import pdfplumber
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import unicodedata

logger = logging.getLogger(__name__)


@dataclass
class DocumentFeatures:
    """Data class to store extracted features"""

    direccion_obras: Optional[str] = None
    financiamiento: Optional[str] = None
    plazo_ejecucion: Optional[str] = None
    presupuesto_oficial: Optional[str] = None
    visita_terreno: Optional[str] = None
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "direccion_obras": self.direccion_obras,
            "financiamiento": self.financiamiento,
            "plazo_ejecucion": self.plazo_ejecucion,
            "presupuesto_oficial": self.presupuesto_oficial,
            "visita_terreno": self.visita_terreno,
            "metadata": self.metadata or {},
        }


class PDFProcessor:
    """Process PDF documents to extract specific features"""

    def __init__(self):
        # Compile regex patterns for faster matching
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns for efficiency"""

        # Dirección de las Obras patterns
        self.direccion_patterns = [
            re.compile(
                r"direcci[oó]n\s+(?:de\s+)?las?\s+obras?[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"ubicaci[oó]n\s+(?:de\s+)?las?\s+obras?[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"lugar\s+(?:de\s+)?ejecuci[oó]n[:\s]+(.+?)(?=\n|\.|;|$)", re.IGNORECASE
            ),
            re.compile(
                r"sitio\s+(?:de\s+)?las?\s+obras?[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
        ]

        # Financiamiento patterns
        self.financiamiento_patterns = [
            re.compile(r"financiamiento[:\s]+(.+?)(?=\n|\.|;|$)", re.IGNORECASE),
            re.compile(
                r"fuente\s+(?:de\s+)?financiamiento[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"recursos\s+(?:de\s+)?financiamiento[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"presupuesto\s+asignado[:\s]+(.+?)(?=\n|\.|;|$)", re.IGNORECASE
            ),
            re.compile(
                r"monto\s+(?:de\s+)?financiamiento[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
        ]

        # Plazo para la Ejecución patterns
        self.plazo_patterns = [
            re.compile(
                r"plazo\s+(?:para\s+)?la?\s+ejecuci[oó]n[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"tiempo\s+(?:de\s+)?ejecuci[oó]n[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"dur[ao]ci[oó]n\s+(?:de\s+)?las?\s+obras?[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"plazo\s+(?:de\s+)?obra[:\s]+(.+?)(?=\n|\.|;|$)", re.IGNORECASE
            ),
            re.compile(
                r"(\d+\s+(?:d[ií]as?|meses?|semanas?|a[nñ]os?))\s+(?:para\s+)?ejecuci[oó]n",
                re.IGNORECASE,
            ),
        ]

        # Presupuesto Oficial patterns
        self.presupuesto_patterns = [
            re.compile(
                r"presupuesto\s+(?:oficial|base|referencial)[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"monto\s+(?:oficial|base|referencial)[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"valor\s+(?:oficial|base|referencial)[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(\$?\s*\d[\d.,]*\s*(?:UF|UTM|CLP|USD|EUR|pesos?))\s+(?:de\s+)?presupuesto",
                re.IGNORECASE,
            ),
            re.compile(
                r"presupuesto\s+(?:de\s+)?(\$?\s*\d[\d.,]*\s*(?:UF|UTM|CLP|USD|EUR|pesos?))",
                re.IGNORECASE,
            ),
        ]

        # Visita a Terreno patterns
        self.visita_patterns = [
            re.compile(
                r"visita\s+(?:a\s+)?terreno[:\s]+(.+?)(?=\n|\.|;|$)", re.IGNORECASE
            ),
            re.compile(
                r"inspecci[oó]n\s+(?:de\s+)?terreno[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"recorrido\s+(?:de\s+)?terreno[:\s]+(.+?)(?=\n|\.|;|$)", re.IGNORECASE
            ),
            re.compile(
                r"fecha\s+(?:de\s+)?visita\s+(?:a\s+)?terreno[:\s]+(.+?)(?=\n|\.|;|$)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4})\s+(?:para\s+)?visita\s+(?:a\s+)?terreno",
                re.IGNORECASE,
            ),
        ]

        # Currency and date patterns for validation
        self.currency_pattern = re.compile(
            r"\$?\s*\d[\d.,]*\s*(?:UF|UTM|CLP|USD|EUR|pesos?|millones|mill[oó]n)",
            re.IGNORECASE,
        )
        self.date_pattern = re.compile(
            r"\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}|\d{1,2}\s+(?:de\s+)?(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?\d{4}",
            re.IGNORECASE,
        )
        self.time_pattern = re.compile(
            r"\d+\s+(?:d[ií]as?|meses?|semanas?|a[nñ]os?)(?:\s+\d+\s+(?:d[ií]as?|meses?|semanas?))?",
            re.IGNORECASE,
        )

    def normalize_text(self, text: str) -> str:
        """Normalize text for better matching"""
        # Convert to lowercase and normalize unicode
        text = text.lower()
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ASCII", "ignore")
            .decode("ASCII")
        )
        # Remove extra whitespace
        text = " ".join(text.split())
        return text

    def clean_extracted_value(self, value: str) -> str:
        """Clean extracted value"""
        value = value.strip()
        # Remove common trailing punctuation
        value = re.sub(r"^[:;\s]+|[:;\s]+$", "", value)
        return value

    def extract_with_patterns(
        self, text: str, patterns: List[re.Pattern]
    ) -> Optional[str]:
        """Extract value using multiple patterns"""
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                # Get the longest match (often more complete)
                matches = [self.clean_extracted_value(m) for m in matches if m.strip()]
                if matches:
                    # Prioritize matches with currency, dates, or longer text
                    for match in sorted(
                        matches,
                        key=lambda x: (
                            bool(
                                self.currency_pattern.search(x)
                                or self.date_pattern.search(x)
                                or self.time_pattern.search(x)
                            ),
                            len(x),
                        ),
                        reverse=True,
                    ):
                        if len(match) > 3:  # Minimum length
                            return match
        return None

    def search_nearby_text(
        self, text: str, keyword: str, window: int = 200
    ) -> Optional[str]:
        """Search for keyword and extract nearby text"""
        idx = text.lower().find(keyword.lower())
        if idx != -1:
            start = max(0, idx - window)
            end = min(len(text), idx + len(keyword) + window)
            return text[start:end]
        return None

    def extract_direccion_obras(self, text: str) -> Optional[str]:
        """Extract Dirección de las Obras"""
        result = self.extract_with_patterns(text, self.direccion_patterns)
        if result:
            return result

        # Fallback: Search for address-like patterns
        address_patterns = [
            r"calle\s+.+?\s+\d+",
            r"av\.?\s+.+?\s+\d+",
            r"avenida\s+.+?\s+\d+",
            r"comuna\s+.+?,\s+.+",
            r"regi[oó]n\s+.+?,\s+.+",
        ]

        for pattern in address_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0]

        return None

    def extract_financiamiento(self, text: str) -> Optional[str]:
        """Extract Financiamiento information"""
        result = self.extract_with_patterns(text, self.financiamiento_patterns)
        if result:
            return result

        # Fallback: Look for funding-related terms near currency amounts
        funding_keywords = ["fondo", "recursos", "asignación", "subsidio", "aportes"]
        for keyword in funding_keywords:
            context = self.search_nearby_text(text, keyword)
            if context:
                # Extract currency amounts from context
                amounts = self.currency_pattern.findall(context)
                if amounts:
                    return f"{keyword}: {', '.join(amounts)}"

        return None

    def extract_plazo_ejecucion(self, text: str) -> Optional[str]:
        """Extract Plazo para la Ejecución de las Obras"""
        result = self.extract_with_patterns(text, self.plazo_patterns)
        if result:
            return result

        # Fallback: Look for time periods
        time_matches = self.time_pattern.findall(text)
        if time_matches:
            # Filter plausible time periods (not too short or too long)
            plausible_periods = []
            for period in time_matches:
                # Check if it's a reasonable construction period
                # Typically between 30 days and 5 years
                numbers = re.findall(r"\d+", period)
                if numbers:
                    num = int(numbers[0])
                    if "día" in period.lower() and 30 <= num <= 365 * 2:
                        plausible_periods.append(period)
                    elif "mes" in period.lower() and 1 <= num <= 60:
                        plausible_periods.append(period)
                    elif "año" in period.lower() and 1 <= num <= 5:
                        plausible_periods.append(period)

            if plausible_periods:
                return plausible_periods[0]

        return None

    def extract_presupuesto_oficial(self, text: str) -> Optional[str]:
        """Extract Presupuesto Oficial"""
        result = self.extract_with_patterns(text, self.presupuesto_patterns)
        if result:
            return result

        # Fallback: Look for large currency amounts
        amounts = self.currency_pattern.findall(text)
        if amounts:
            # Filter for likely budget amounts (typically larger)
            # Look for amounts with "millones" or without decimal places for thousands
            for amount in amounts:
                if "millón" in amount.lower() or "millones" in amount.lower():
                    return amount
                # Check if it's a large number (likely budget)
                numbers = re.findall(r"[\d,.]+", amount)
                if numbers:
                    # Remove dots (thousand separators) and commas (decimal separators)
                    num_str = numbers[0].replace(".", "").replace(",", ".")
                    try:
                        num = float(num_str)
                        if num > 100000:  # Likely a budget amount
                            return amount
                    except ValueError:
                        continue

        return None

    def extract_visita_terreno(self, text: str) -> Optional[str]:
        """Extract Visita a Terreno information"""
        result = self.extract_with_patterns(text, self.visita_patterns)
        if result:
            return result

        # Fallback: Look for dates near "visita" or "terreno"
        for keyword in ["visita", "terreno", "inspección"]:
            context = self.search_nearby_text(text, keyword)
            if context:
                dates = self.date_pattern.findall(context)
                if dates:
                    return f"{keyword}: {dates[0]}"

        # Look for "no se requiere visita" or similar
        no_visita_patterns = [
            r"no\s+(?:se\s+)?requiere\s+visita",
            r"sin\s+visita\s+(?:a\s+)?terreno",
            r"visita\s+(?:a\s+)?terreno\s+no\s+requerida",
        ]

        for pattern in no_visita_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "No requerida"

        return None

    def extract_from_pdf(self, pdf_path: Path) -> Optional[DocumentFeatures]:
        """Extract all features from a PDF file"""
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return None

        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract text from first few pages (most relevant info usually at beginning)
                text_parts = []
                metadata = {
                    "total_pages": len(pdf.pages),
                    "file_size": pdf_path.stat().st_size,
                    "file_name": pdf_path.name,
                }

                # Read first 10 pages or all pages if less
                pages_to_read = min(10, len(pdf.pages))
                for i in range(pages_to_read):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)

                # If we didn't get much text from first pages, try all pages
                if len("".join(text_parts)) < 1000 and len(pdf.pages) > 10:
                    logger.info(
                        f"Little text in first {pages_to_read} pages, scanning all pages"
                    )
                    text_parts = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)

                full_text = "\n".join(text_parts)

                if not full_text or len(full_text.strip()) < 100:
                    logger.warning(
                        f"PDF appears to be scanned or has little text: {pdf_path}"
                    )

                    # Try to extract tables as fallback
                    tables_data = []
                    for i, page in enumerate(pdf.pages[:5]):  # First 5 pages
                        tables = page.extract_tables()
                        for table in tables:
                            tables_data.append(str(table))

                    if tables_data:
                        full_text = "\n".join(tables_data)
                        metadata["extraction_method"] = "table_extraction"
                    else:
                        metadata["extraction_method"] = "failed"
                        return DocumentFeatures(metadata=metadata)
                else:
                    metadata["extraction_method"] = "text_extraction"
                    metadata["text_length"] = len(full_text)

                # Extract features
                features = DocumentFeatures(
                    direccion_obras=self.extract_direccion_obras(full_text),
                    financiamiento=self.extract_financiamiento(full_text),
                    plazo_ejecucion=self.extract_plazo_ejecucion(full_text),
                    presupuesto_oficial=self.extract_presupuesto_oficial(full_text),
                    visita_terreno=self.extract_visita_terreno(full_text),
                    metadata=metadata,
                )

                # Log extraction results
                extracted_count = sum(
                    1
                    for value in [
                        features.direccion_obras,
                        features.financiamiento,
                        features.plazo_ejecucion,
                        features.presupuesto_oficial,
                        features.visita_terreno,
                    ]
                    if value is not None
                )

                logger.info(
                    f"Extracted {extracted_count}/5 features from {pdf_path.name}"
                )

                return features

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            return DocumentFeatures(metadata={"error": str(e)})

    def batch_process_pdfs(
        self, pdf_paths: List[Path], output_file: Optional[Path] = None
    ) -> List[Dict]:
        """Process multiple PDFs and optionally save results"""
        results = []

        for pdf_path in pdf_paths:
            logger.info(f"Processing: {pdf_path.name}")
            features = self.extract_from_pdf(pdf_path)

            if features:
                result = {
                    "file_name": pdf_path.name,
                    "file_path": str(pdf_path),
                    **features.to_dict(),
                }
                results.append(result)

                # Log extraction summary
                extracted = [
                    k for k, v in features.to_dict().items() if v and k != "metadata"
                ]
                if extracted:
                    logger.info(f"  Extracted: {', '.join(extracted)}")
                else:
                    logger.warning(f"  No features extracted")

        # Save results if output file specified
        if output_file and results:
            self.save_results(results, output_file)

        return results

    def save_results(self, results: List[Dict], output_file: Path):
        """Save results to CSV or JSON"""
        import json
        import csv

        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.suffix == ".json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_file}")

        elif output_file.suffix == ".csv":
            # Flatten the dictionary for CSV
            flattened_results = []
            for result in results:
                flat = result.copy()
                if "metadata" in flat:
                    metadata = flat.pop("metadata")
                    for k, v in metadata.items():
                        flat[f"metadata_{k}"] = str(v)
                flattened_results.append(flat)

            if flattened_results:
                fieldnames = flattened_results[0].keys()
                with open(output_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(flattened_results)
                logger.info(f"Results saved to {output_file}")


async def process_downloaded_documents(document_paths: List[str]):
    """Process all downloaded documents"""
    logger.info(f"Processing {len(document_paths)} downloaded documents")

    processor = PDFProcessor()
    results = []

    for doc_path in document_paths:
        path = Path(doc_path)
        if path.exists():
            features = processor.extract_from_pdf(path)
            if features:
                results.append({"document": path.name, **features.to_dict()})

    # Save results
    output_dir = Path("extracted_features")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"features_{timestamp}.json"

    processor.save_results(results, output_file)

    # Generate summary
    if results:
        total_docs = len(results)
        extracted_counts = {
            feature: 0
            for feature in [
                "direccion_obras",
                "financiamiento",
                "plazo_ejecucion",
                "presupuesto_oficial",
                "visita_terreno",
            ]
        }

        for result in results:
            for feature in extracted_counts:
                if result.get(feature):
                    extracted_counts[feature] += 1

        logger.info("\n" + "=" * 50)
        logger.info("FEATURE EXTRACTION SUMMARY:")
        logger.info(f"Total documents processed: {total_docs}")
        for feature, count in extracted_counts.items():
            percentage = (count / total_docs * 100) if total_docs > 0 else 0
            logger.info(
                f"{feature.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
            )
        logger.info("=" * 50)

    return results


# Quick test function
def test_pdf_processing():
    """Test the PDF processor on a sample file"""
    processor = PDFProcessor()

    # Create a test text
    test_text = """
    CONTRATO DE OBRA PÚBLICA
    
    Dirección de las Obras: Calle Principal 123, Comuna de Santiago, Región Metropolitana
    
    Financiamiento: Los recursos provienen del Fondo Nacional de Desarrollo Regional (FNDR) por un monto de $150.000.000 (pesos chilenos).
    
    Plazo para la Ejecución de las Obras: 180 días calendario, contados desde la fecha de inicio.
    
    Presupuesto Oficial: $250.000.000 (doscientos cincuenta millones de pesos), incluyendo IVA.
    
    Visita a Terreno: Se realizará el día 15/03/2024 a las 10:00 hrs. La visita es obligatoria para todos los proponentes.
    
    Otros términos y condiciones aplican según lo establecido en las bases de licitación.
    """

    print("Test Extraction Results:")
    print("-" * 50)
    print(f"Dirección: {processor.extract_direccion_obras(test_text)}")
    print(f"Financiamiento: {processor.extract_financiamiento(test_text)}")
    print(f"Plazo Ejecución: {processor.extract_plazo_ejecucion(test_text)}")
    print(f"Presupuesto: {processor.extract_presupuesto_oficial(test_text)}")
    print(f"Visita Terreno: {processor.extract_visita_terreno(test_text)}")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Test the processor
    test_pdf_processing()

    # Example usage:
    # processor = PDFProcessor()
    # features = processor.extract_from_pdf(Path("document.pdf"))
    # if features:
    #     print(features.to_dict())
