import asyncio
import aiohttp
import aiofiles
import difflib
import csv
from difflib import SequenceMatcher
import logging
import re
import json
import os
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set

import hashlib
import random

import pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

FIELDS = [
    "IDLicitacion",
    "NombreLicitacion",
    "Region",
    "Fecha de Publicacion",
    "Fecha de Cierre",
    "Especialidad",
    "Sub-Especialidad",
    "Categoría",
    "Dirección de las Obras",
    "Financiamiento",
    "Plazo para la Ejecución de las Obra",
    "Presupuesto Oficial",
    "Visita a Terreno",
]


def to_csv(json_data, csv_filepath: str) -> None:
    if not json_data:
        print("No data to convert")
        return

    # Flatten the data by extracting only non-nested, non-dict, non-list values
    flat_data = []
    for item in json_data:
        flat_item = {}
        for key, value in item.items():
            # Skip nested structures (dicts and lists)
            if not isinstance(value, (dict, list)):
                flat_item[key] = value
        flat_data.append(flat_item)

    # Get fieldnames from the first item (if available)
    fieldnames = list(flat_data[0].keys()) if flat_data else []

    with open(csv_filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_data)


def fuzzy_match_strings(
    text1: str,
    text2: str,
    threshold: float = 0.8,
) -> float:
    """
    Calculate similarity between two strings.

    Args:
        text1: First string
        text2: Second string
        threshold: Minimum similarity score (0-1)
        use_levenshtein: Use Levenshtein distance if True, else use SequenceMatcher

    Returns:
        Similarity score (0-1)
    """

    # Clean and normalize strings
    def normalize(text: str) -> str:
        # Convert to lowercase, remove extra spaces and punctuation
        text = text.lower()
        text = re.sub(r"[^\w\sáéíóúñü]", "", text)  # Keep Spanish characters
        text = re.sub(r"\s+", " ", text).strip()
        return text

    clean1 = normalize(text1)
    clean2 = normalize(text2)

    # Exact match after normalization
    if clean1 == clean2:
        return 1.0

    # Check for partial matches (substring)
    if clean1 in clean2 or clean2 in clean1:
        return 0.9  # High score for substring matches

    return SequenceMatcher(None, clean1, clean2).ratio()


class RateLimiter:
    """Rate limiter with semaphore support"""

    def __init__(self, rate_limit: float = 1.0, max_concurrent: int = 10):
        self.rate_limit = rate_limit
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.last_request_time = 0

    async def __aenter__(self):
        await self.semaphore.acquire()
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_request_time
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_request_time = asyncio.get_event_loop().time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.semaphore.release()


class CacheManager:
    """Cache manager for web pages and documents"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.web_cache_dir = self.cache_dir / "web_pages"
        self.doc_cache_dir = self.cache_dir / "documents"
        self.setup_dirs()

    def setup_dirs(self):
        """Create cache directories if they don't exist"""
        self.web_cache_dir.mkdir(parents=True, exist_ok=True)
        self.doc_cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, url: str, cache_type: str = "web") -> Path:
        """Generate cache file path from URL"""
        # Create a filename-safe hash of the URL
        url_hash = hashlib.md5(url.encode()).hexdigest()

        # Extract extension if present in URL
        parsed = urlparse(url)
        path = parsed.path
        ext = ""
        if "." in path:
            ext = Path(path).suffix
            if not ext or len(ext) > 10:  # Sanity check
                ext = ".html" if cache_type == "web" else ".bin"
        else:
            ext = ".html" if cache_type == "web" else ".bin"

        if cache_type == "web":
            return self.web_cache_dir / f"{url_hash}{ext}"
        else:
            return self.doc_cache_dir / f"{url_hash}{ext}"

    async def save_to_cache(self, url: str, content: bytes, cache_type: str = "web"):
        """Save content to cache"""
        cache_path = self.get_cache_path(url, cache_type)
        async with aiofiles.open(cache_path, "wb") as f:
            await f.write(content)
        logger.debug(f"Cached {url} to {cache_path}")

    async def load_from_cache(
        self, url: str, cache_type: str = "web"
    ) -> Optional[bytes]:
        """Load content from cache if exists"""
        cache_path = self.get_cache_path(url, cache_type)
        if cache_path.exists():
            async with aiofiles.open(cache_path, "rb") as f:
                content = await f.read()
            logger.debug(f"Loaded {url} from cache")
            return content
        return None

    def is_cached(self, url: str, cache_type: str = "web") -> bool:
        """Check if URL is cached"""
        cache_path = self.get_cache_path(url, cache_type)
        return cache_path.exists()


class MercadoPublicoScraper:
    """Main scraper class for MercadoPublico.cl"""

    def __init__(self, base_url: str = "https://www.mercadopublico.cl"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache = CacheManager()
        self.rate_limiter = RateLimiter(rate_limit=1.0, max_concurrent=10)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

        self.current_data = []
        self._regions = []

    async def __aenter__(self):
        await self.init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def init_session(self):
        """Initialize aiohttp session with cookies"""
        self.session = aiohttp.ClientSession(
            headers=self.headers, cookie_jar=aiohttp.CookieJar()
        )

    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

    async def fetch(
        self,
        url: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        use_cache: bool = True,
    ) -> Optional[bytes]:
        """Fetch URL with rate limiting and caching"""
        if not self.session:
            raise RuntimeError(
                "Session not initialized. Use async with or call init_session()"
            )

        # Check cache first
        if use_cache and method == "GET":
            cached = await self.cache.load_from_cache(url)
            if cached:
                return cached

        async with self.rate_limiter:
            try:
                if method == "GET":
                    async with self.session.get(url) as response:
                        content = await response.read()
                elif method == "POST":
                    async with self.session.post(url, json=data) as response:
                        content = await response.read()
                else:
                    raise ValueError(f"Unsupported method: {method}")

                # Save to cache
                if use_cache and response.status == 200:
                    await self.cache.save_to_cache(url, content)

                logger.info(f"Fetched {url} - Status: {response.status}")
                return content if response.status == 200 else None

            except Exception as e:
                logger.error(f"Error fetching {url}: {e}")
                return None

    async def get_initial_session(self) -> bool:
        """Step 1: Get initial session cookie"""
        url = f"{self.base_url}/Home/BusquedaLicitacion"
        result = await self.fetch(url, use_cache=True)
        return result is not None

    async def search_licitaciones(self, page: int = 1) -> Optional[bytes]:
        """Step 2: Search licitaciones"""
        url = f"{self.base_url}/BuscarLicitacion/Home/Buscar"

        payload = {
            "textoBusqueda": "",
            "idEstado": "5",
            "codigoRegion": "-1",
            "idTipoLicitacion": "2",
            "fechaInicio": None,
            "fechaFin": None,
            "registrosPorPagina": "10",
            "idTipoFecha": [],
            "idOrden": "1",
            "compradores": [],
            "garantias": None,
            "rubros": [],
            "proveedores": [],
            "montoEstimadoTipo": [0],
            "esPublicoMontoEstimado": ["1"],
            "pagina": page,
        }

        return await self.fetch(url, method="POST", data=payload)

    def extract_licitacion_links(self, html_content: bytes) -> List[str]:
        """Step 3: Extract licitacion links from search results"""
        soup = BeautifulSoup(html_content, "html.parser")
        links = []

        # Find all a tags with the pattern
        pattern = re.compile(r"\$\.Busqueda\.verFicha\('")

        for a_tag in soup.find_all("a"):
            if "onclick" not in a_tag.attrs:
                continue
            href = a_tag["onclick"]
            if pattern.search(href):
                # Extract the URL from the JavaScript call
                match = re.search(r"'(http?://[^']+)'", href)
                if match:
                    links.append(match.group(1))

        for item in soup.findAll(class_="licitacion-publicada"):
            identifier = (
                item.find(class_="id-licitacion").find(class_="clearfix").text.strip()
            )
            name = item.find("h2").text.strip()
            highlight = item.findAll(class_="highlight-text text-weight-light")
            state = 0
            patt = "[0-9]+/[0-9]+/[0-9]+"
            pub_date = None
            close_date = None
            for hi in highlight:
                text = hi.text
                if re.match(patt, text):
                    if state == 0:
                        pub_date = text
                        state += 1
                    elif state == 1:
                        close_date = text
                        break
            self.current_data.append(
                {
                    "IDLicitacion": identifier,
                    "NombreLicitacion": name,
                    "Fecha de Publicacion": pub_date,
                    "Fecha de Cierre": close_date,
                }
            )

        logger.info(f"Extracted {len(links)} licitacion links")
        return links

    async def get_licitacion_details(self, url: str) -> Optional[bytes]:
        """Step 4: Get licitacion detail page"""
        return await self.fetch(url, use_cache=True)

    def extract_attachment_url(self, html_content: bytes, idx: int) -> Optional[str]:
        """Step 6: Extract attachment URL from imgAdjuntos element"""
        soup = BeautifulSoup(html_content, "html.parser")
        img_element = soup.find(id="imgAdjuntos")

        att_base_url = "https://www.mercadopublico.cl/Procurement/Modules/"

        self.current_data[idx]["Region"] = soup.find(id="lblFicha2Region").text

        if img_element and img_element.get("onclick"):
            src = img_element["onclick"]
            # Extract URL from JavaScript open() call
            match = re.search(r"open\('([^']+)'", src)
            if match:
                # Convert relative URL to absolute
                relative_url = match.group(1)
                if relative_url.startswith("../"):
                    relative_url = relative_url[3:]
                return urljoin(att_base_url, relative_url)

        return None

    async def get_attachment_page(self, url: str) -> Optional[bytes]:
        """Step 7: Get attachment page"""
        return await self.fetch(url, use_cache=True)

    def extract_form_data(self, html_content: bytes) -> Optional[Dict]:
        """Extract form data from attachment page for 'anexos complementarios' rows only"""
        soup = BeautifulSoup(html_content, "html.parser")
        form_data = {}

        form = soup.find(id="DWNL_grdId")
        file_name = ""
        file_size = None
        match = None

        input_imgs = []
        for row in form.findAll("tr"):
            columns = row.find_all("td")
            if len(columns) < 7:
                continue
            file_name = columns[1].text.lower()
            file_size = columns[4].text
            input_img = columns[6]
            match = fuzzy_match_strings("anexo_complementario", file_name)
            if match > 0.1:
                input_imgs.append(input_img)
            else:
                if "comp" in file_name:
                    logger.warning("Possible match %s", file_name)

        logger.info(
            f"Processing row - File: {file_name[:60]}, "
            f"Size: {file_size}, Similarity: {match:.2f}"
        )

        form_base_url = "https://www.mercadopublico.cl/Procurement/Modules/Attachment/ViewAttachment.aspx"

        if form:
            # another form
            form_data["action"] = urljoin(
                form_base_url, soup.find("form").get("action", "")
            )

            # Extract all input fields
            inputs = {}
            for input_tag in soup.find_all("input"):
                name = input_tag.get("name")
                value = input_tag.get("value", "")
                if name:
                    inputs[name] = value

            form_data["inputs"] = inputs

            # Find image input (submit button)
            if input_imgs:
                latest = input_imgs[0].find("input")
                if latest and "name" in latest.attrs:
                    form_data["image_input_name"] = latest["name"]
            else:
                logger.warning("No file found")
                return None

            # Add metadata about the file
            if file_name and file_size:
                form_data["file_name"] = columns[1].get_text(strip=True)
                form_data["similarity_score"] = match

                if len(columns) >= 5:
                    form_data["file_size"] = columns[4].get_text(strip=True)

            logger.info(f"Extracted form for: {form_data.get('file_name', 'Unknown')}")
            return form_data
        else:
            logger.warning("No form found in the matching row")
            return None

    async def download_document(
        self,
        form_action: str,
        form_data: Dict,
    ) -> Optional[str]:
        """Step 8/9: Submit form to download document with retries and chunked download"""
        if not self.session:
            return None

        # Generate a unique cache URL for this document
        cache_url = f"{form_action}_{hashlib.md5(str(form_data).encode()).hexdigest()}"
        cache_path = self.cache.get_cache_path(cache_url, cache_type="doc")

        # Check if already cached
        if cache_path.exists():
            logger.info(f"Document already cached: {cache_path}")
            return str(cache_path)

        # Prepare form data for submission
        data = aiohttp.FormData()
        for key, value in form_data["inputs"].items():
            data.add_field(key, value)

        # Add coordinates for image input click simulation
        if "image_input_name" in form_data and form_data["image_input_name"]:
            # Simulate a click at position (1, 1)
            data.add_field(f"{form_data['image_input_name']}.x", "1")
            data.add_field(f"{form_data['image_input_name']}.y", "1")

        # Retry configuration
        max_retries = 3
        retry_delays = [2, 5, 10]  # Exponential backoff in seconds
        last_exception = None

        for attempt in range(max_retries):
            try:
                async with self.rate_limiter:
                    async with self.session.post(
                        form_action,
                        data=data,
                        timeout=aiohttp.ClientTimeout(
                            total=300
                        ),  # 5 minute timeout for large files
                    ) as response:

                        if response.status == 200:
                            # Get content disposition to determine filename
                            content_disposition = response.headers.get(
                                "Content-Disposition", ""
                            )
                            filename = None
                            if "filename=" in content_disposition:
                                # Extract filename from Content-Disposition header
                                filename_match = re.search(
                                    r'filename="([^"]+)"', content_disposition
                                )
                                if filename_match:
                                    filename = filename_match.group(1)

                            # Get content type
                            content_type = response.headers.get("Content-Type", "")
                            content_length = response.headers.get("Content-Length")

                            logger.info(
                                f"Downloading document from {form_action} "
                                f"(Attempt {attempt + 1}/{max_retries}) - "
                                f"Size: {content_length or 'Unknown'} bytes, "
                                f"Type: {content_type}"
                            )

                            # Chunked download with progress reporting
                            total_bytes = 0
                            chunk_size = 8192 * 1024  # 8MB chunks for large documents

                            async with aiofiles.open(cache_path, "wb") as f:
                                async for chunk in response.content.iter_chunked(
                                    chunk_size
                                ):
                                    await f.write(chunk)
                                    total_bytes += len(chunk)

                                    # Log progress for large files
                                    if (
                                        content_length
                                        and int(content_length) > 10 * 1024 * 1024
                                    ):  # > 10MB
                                        if (
                                            total_bytes % (10 * 1024 * 1024) == 0
                                        ):  # Every 10MB
                                            progress = (
                                                total_bytes / int(content_length)
                                            ) * 100
                                            logger.info(
                                                f"Download progress: {total_bytes / (1024*1024):.1f}MB / "
                                                f"{int(content_length) / (1024*1024):.1f}MB "
                                                f"({progress:.1f}%)"
                                            )

                            # Verify the file was written successfully
                            if not cache_path.exists():
                                raise IOError(
                                    f"Cache file was not created: {cache_path}"
                                )

                            file_size = cache_path.stat().st_size

                            # Validate minimum file size (should be more than just headers)
                            if file_size < 100:  # Less than 100 bytes is suspicious
                                logger.warning(
                                    f"Document seems very small ({file_size} bytes). "
                                    f"It might be an error page."
                                )
                                # Read first 500 bytes to check if it's HTML/error
                                async with aiofiles.open(cache_path, "rb") as f:
                                    first_bytes = await f.read(500)
                                    if (
                                        b"<html" in first_bytes.lower()
                                        or b"error" in first_bytes.lower()
                                    ):
                                        logger.error(
                                            "Downloaded content appears to be an error page"
                                        )
                                        cache_path.unlink()  # Remove the invalid file
                                        raise ValueError(
                                            "Downloaded content is an error page"
                                        )

                            logger.info(
                                f"Successfully downloaded document to {cache_path} "
                                f"({file_size / (1024*1024):.2f} MB)"
                            )

                            # If we have a proper filename from Content-Disposition,
                            # create a symbolic link for easier access
                            if filename:
                                safe_filename = re.sub(r"[^\w\-_.]", "_", filename)
                                symlink_path = cache_path.parent / safe_filename
                                try:
                                    if symlink_path.exists():
                                        symlink_path.unlink()
                                    symlink_path.symlink_to(cache_path.name)
                                    logger.debug(
                                        f"Created symlink: {symlink_path} -> {cache_path.name}"
                                    )
                                except (OSError, NotImplementedError) as e:
                                    logger.debug(f"Could not create symlink: {e}")

                            return str(cache_path)

                        elif response.status in [
                            429,
                            503,
                        ]:  # Rate limited or service unavailable
                            logger.warning(
                                f"Rate limited (status {response.status}) on attempt {attempt + 1}. "
                                f"Retrying in {retry_delays[attempt]} seconds..."
                            )
                            await asyncio.sleep(retry_delays[attempt])
                            continue

                        elif response.status in [404, 403, 401]:  # Permanent errors
                            logger.error(
                                f"Failed to download document: HTTP {response.status} - "
                                f"{response.reason}"
                            )
                            return None

                        else:  # Other HTTP errors
                            logger.warning(
                                f"HTTP error {response.status} on attempt {attempt + 1}. "
                                f"Retrying in {retry_delays[attempt]} seconds..."
                            )
                            await asyncio.sleep(retry_delays[attempt])

            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout downloading document on attempt {attempt + 1}. "
                    f"Retrying in {retry_delays[attempt]} seconds..."
                )
                await asyncio.sleep(retry_delays[attempt])
                last_exception = "TimeoutError"

            except aiohttp.ClientError as e:
                logger.warning(
                    f"Network error on attempt {attempt + 1}: {e}. "
                    f"Retrying in {retry_delays[attempt]} seconds..."
                )
                await asyncio.sleep(retry_delays[attempt])
                last_exception = str(e)

            except IOError as e:
                logger.error(f"File system error: {e}")
                return None

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                await asyncio.sleep(retry_delays[attempt])
                last_exception = str(e)

        # All retries failed
        logger.error(
            f"Failed to download document after {max_retries} attempts. "
            f"Last error: {last_exception}"
        )

        # Clean up any partially downloaded file
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.debug(f"Cleaned up incomplete download: {cache_path}")
            except OSError:
                pass

        return None

    def save_data(self):
        with open("results.json", "w") as file:
            file.write(json.dumps(self.current_data))

    async def scrape_all_licitaciones(
        self, max_pages: Optional[int] = None
    ) -> List[str]:
        """Main scraping orchestration"""
        logger.info("Starting MercadoPublico scraper...")

        # Step 1: Get initial session
        logger.info("Step 1: Getting initial session...")
        if not await self.get_initial_session():
            logger.error("Failed to get initial session")
            return []

        all_licitacion_links = []
        page = 1

        # Step 2: Search and collect all licitaciones
        logger.info("Step 2: Searching licitaciones...")
        while True:
            if max_pages and page > max_pages:
                logger.info(f"Reached max pages limit: {max_pages}")
                break

            logger.info(f"Fetching page {page}...")
            search_result = await self.search_licitaciones(page)

            if not search_result:
                logger.warning(f"No results for page {page}, stopping")
                break

            # Check if page has results
            soup = BeautifulSoup(search_result, "html.parser")
            if "No hay licitaciones" in soup.text:
                logger.info("No more licitaciones found")
                break

            # Step 3: Extract links
            page_links = self.extract_licitacion_links(search_result)

            self.save_data()
            if not page_links:
                logger.info("No more links found on this page")
                break

            all_licitacion_links.extend(page_links)
            logger.info(
                f"Page {page}: Found {len(page_links)} links, Total: {len(all_licitacion_links)}"
            )

            page += 1
            # Small delay between pages
            await asyncio.sleep(0.5)

        logger.info(f"Total licitacion links found: {len(all_licitacion_links)}")

        # Step 4 & 5: Get all licitacion detail pages
        logger.info("Steps 4-5: Fetching licitacion detail pages...")
        detail_tasks = []
        for link in all_licitacion_links:
            task = self.get_licitacion_details(link)
            detail_tasks.append(task)

        detail_results = await asyncio.gather(*detail_tasks, return_exceptions=True)

        # Step 6: Extract attachment URLs
        logger.info("Step 6: Extracting attachment URLs...")
        attachment_urls = []
        for i, result in enumerate(detail_results):
            if result and not isinstance(result, Exception):
                attachment_url = self.extract_attachment_url(result, i)
                if attachment_url:
                    attachment_urls.append(attachment_url)

        logger.info(f"Found {len(attachment_urls)} attachment URLs")

        # Step 7: Get attachment pages
        logger.info("Step 7: Fetching attachment pages...")
        attachment_tasks = []
        for url in attachment_urls:
            task = self.get_attachment_page(url)
            attachment_tasks.append(task)

        attachment_results = await asyncio.gather(
            *attachment_tasks, return_exceptions=True
        )

        self.save_data()

        # Steps 8-9: Download documents
        logger.info("Steps 8-9: Downloading documents...")
        download_tasks = []
        downloaded_docs = []

        for i, result in enumerate(attachment_results):
            if result and not isinstance(result, Exception):
                form_data = self.extract_form_data(result)
                if form_data and "action" in form_data:
                    task = self.download_document(form_data["action"], form_data)
                    download_tasks.append(task)

        # Process downloads with improved batching
        batch_size = 3  # Smaller batch size for memory management
        successful_downloads = 0
        failed_downloads = 0

        for i in range(0, len(download_tasks), batch_size):
            batch = download_tasks[i : i + batch_size]
            try:
                batch_results = await asyncio.gather(*batch, return_exceptions=True)

                for j, result in enumerate(batch_results):
                    task_index = i + j
                    if isinstance(result, Exception):
                        logger.error(f"Download task {task_index + 1} failed: {result}")
                        failed_downloads += 1
                    elif result:
                        downloaded_docs.append(result)
                        successful_downloads += 1
                        logger.info(
                            f"Downloaded document {task_index + 1}/{len(download_tasks)}: {result}"
                        )
                    else:
                        failed_downloads += 1
                        logger.warning(f"Download task {task_index + 1} returned None")

                # Progress report
                total_processed = min(i + batch_size, len(download_tasks))
                logger.info(
                    f"Download progress: {total_processed}/{len(download_tasks)} "
                    f"(Success: {successful_downloads}, Failed: {failed_downloads})"
                )

                # Be extra polite between batches with jitter
                wait_time = 3 + random.uniform(0, 2)  # 3-5 seconds with jitter
                await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                failed_downloads += len(batch)
                await asyncio.sleep(5)  # Longer delay on batch failure

        logger.info(
            f"Download complete. Successful: {successful_downloads}, "
            f"Failed: {failed_downloads}, Total: {len(download_tasks)}"
        )

        results = await pdf.process_downloaded_documents(downloaded_docs)
        for idx, res in enumerate(results):
            self.current_data[idx].update(res)

        self.save_data()

        to_csv(self.current_data, "result.csv")

        logger.info(f"Scraping complete. Downloaded {len(downloaded_docs)} documents.")
        return all_licitacion_links

    async def run_with_progress(self, max_pages: Optional[int] = None):
        """Run scraper with progress reporting"""
        try:
            results = await self.scrape_all_licitaciones(max_pages)
            return results
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return []


async def main():
    """Main entry point"""
    # Configuration
    MAX_PAGES = 1  # Set to None for all pages, or a number for testing
    BASE_URL = "http://www.mercadopublico.cl"

    # Create output directory
    output_dir = Path("scraped_data")
    output_dir.mkdir(exist_ok=True)

    async with MercadoPublicoScraper(base_url=BASE_URL) as scraper:
        logger.info(f"Starting scraping with base URL: {BASE_URL}")
        logger.info(f"Max pages: {MAX_PAGES if MAX_PAGES else 'All'}")

        # Run the scraper
        start_time = datetime.now()
        results = await scraper.run_with_progress(MAX_PAGES)
        end_time = datetime.now()

        # Save results summary
        summary = {
            "base_url": BASE_URL,
            "max_pages": MAX_PAGES,
            "total_licitaciones": len(results),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "licitaciones": results,
        }

        summary_path = output_dir / "scraping_summary.json"
        async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(summary, indent=2, ensure_ascii=False))

        logger.info(
            f"Scraping completed in {(end_time - start_time).total_seconds():.2f} seconds"
        )
        logger.info(f"Summary saved to: {summary_path}")
        logger.info(f"Cache directory: {scraper.cache.cache_dir}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
