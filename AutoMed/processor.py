"""
processor.py — FASTA-to-Feature Translator for TB Drug Resistance Models
=========================================================================
Bridges raw FASTA sequence input and the variant feature names expected by
XGBoost/ML models (e.g., 'katG_Ser315Thr', 'rpoB_Ser450Leu').

Pipeline:
  1. Sequence Validation  — parse_fasta() + validate_sequence()
  2. Variant Extraction   — extract_variants_from_sequence()
  3. Feature Mapping      — get_model_input()

Dependencies:
  pip install biopython
"""

import re
import logging
from typing import Optional

# BioPython for codon translation
try:
    from Bio.Seq import Seq
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    logging.warning(
        "BioPython not installed. Translation will use built-in codon table. "
        "Install with: pip install biopython"
    )

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SECTION 1 — Reference Data
# ---------------------------------------------------------------------------

# Standard genetic code (codon → single-letter AA)
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Single-letter → Three-letter amino acid conversion
AA_1TO3 = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
    "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
    "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
    "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
    "*": "Stop",
}

# ---------------------------------------------------------------------------
# H37Rv reference protein sequences (first ~500 AA shown; extend as needed).
# Source: NCBI NC_000962.3
# These are the *translated* reference proteins used for comparison.
# ---------------------------------------------------------------------------
REFERENCE_PROTEINS = {
    # katG — Rv1908c — catalase-peroxidase (key isoniazid resistance gene)
    "katG": (
        "MTGSAPPHIRDPAAQLAAEGFNLANPDKITNAFLEQNHHEDRFPANRGAVSIDDKALMAEL"
        "EQIVDTVASLAKEDPHGLNQIAFYDKFPPMGLTAIGQLSSFLASRLTHAQEPMFRGLSTYF"
        "ETRWTQEILAATLQGKQHAQPYVGDLIDALPSPGRHGQRQRPNIVPTSRPDSAGFLQNGAL"
        "TAALRSRSPTDAAFRESIRQGLAEFSGLRNEAIQAFGKAQQDLDPFRRAQIQHFGTLGKVAM"
        "HRESWLHQLLFGGLHELIDFRDLRGQRIDIEELLVLVAGKIRDLKRQMDEGDHAVQYVPQD"
        "RLPHAVLARLQQLAVSMPEDFDFRALLSRSTTQEYRQALRSQPAHQLAAAMDDRANLVQYAR"
        "NFHMDKLLPQVEEQLRQAVAALQQLRLAEPRAEVAPTPAQDSGSIRNASAFLRQRRERTGNG"
        "GDGALRPASADPEAQTQQVAATGFNDAASPQALCPAFSTLSASNTLLHSRPALASIPQRQAL"
        "LNQILHDPYRQTMSAVQHRQILSYLSRLPSDSPFRRLLLAAKAEFLSGK"
    ),
    # rpoB — Rv0667 — RNA polymerase beta subunit (rifampicin resistance)
    "rpoB": (
        "MSQNFEHLNADEPDQAYRSLMQSQYNQPSAPQQQPAAQPRPAQQQRPAQQQRPAQQQRAGAP"
        "QHQPAAEDPAKDAPPSLTSQPQRPASRVPQSQIAQPQPAQPQSAANAHQPQTAPQTSMQPA"
        "QHRSQPQSGSQQPQDQGTMDVQFIPVHRSQSQNTQGQANSPQQPVQPQSAPAQATTSIPQT"
        "QNQQAQPYQQQLPQQTPQQQLPQQQVPMAQPQTAPVQPQTAPVQPQTAPVQPQTAPVQPQ"
        "MGSMQQSGPVQRPYRDGPSQSAPPYQPPYSAPPMQYPPPQYAPAQYSAPAPQYQQIQQQLPA"
        "AAQSAPVQPQQLQNNLRDLIAKGKKLQLRGLKEMVTDSQDMPITELVSDLIANRGETAVEEL"
        "RNQIRQMTPDPSNIFQELMAGLDSRSQISKMSDGPSTPTRELAKFALEQMKAASDIAAATRE"
        "AIERLESLDNRQIFVDDLKTQMEGDIEELRKRMGIGELKDAAQLMRELSSRSGQINPHLALL"
        "DQTAELAERLKQQLSEGQPKNPQLNSPQSQQPQRAQPALMPQGQRQAPVIQPTSQQLLQQN"
        "QAQLPQMQQSALTQSQQLQQQPQQAPPQAQNQPVLPAPVQASPAQAQPQLPQPQVQPQLPQ"
    ),
    # inhA — Rv1484 — enoyl-ACP reductase (isoniazid/ethionamide resistance)
    "inhA": (
        "MQSGHTFLAGKTIEEALERAGVDVMILNASAGVGHELGGASRKLSEMIDRFFGTDVIVNNAS"
        "SVGMYGSRDELREQQKNNVSRQIMDRIQAEIGVDIVLDLVLPALLSGRPKSVTLGDIAGMG"
        "HTGSAALQYADLGFSPDVPQIVDGELAHLERQAAQHGITREQFDAMSALVYDSHPNGLVHGD"
        "DEQHLLADLMKQHGIAPEDVQTLSDAEQLATQKQWRKLAQEAGIAALPYIAKDYAMAHGQD"
        "LKTATVLIQMLDQNLEKGEQVVAAIQMLAENNMKIYGPETAVQGVEDSPSHEQSALEEEGK"
    ),
    # embB — Rv3795 — arabinosyltransferase B (ethambutol resistance)
    "embB": (
        "MSNKKHPFSHAFYQNPRDRQRFKGAIISALTLFLVSAPMAQDQAAPQPVQVSTAAPDRTPQ"
        "AQVLPQTPAPAQAPVAQNQTPAPAPQTPASAQVAPQTPTTAPAAPQTPVSAQTPTPAPAQT"
        "PATAQAPQTPASAQVAPQTPTTAPAAPQTPVSAQTPTPAPAQTPATAQAPQTPASSAQVTPQ"
        "APRTPASAQTPVSAQAPRTPASAKVTPEAPRTPASAQTPTAAPAPKTPVTAQVAAAQTSVAQ"
        "ALPQTAASAQTPATAQAPQAPASAQVTPQAPRTPASAQTPVSAQAPRTPASAKVTPEAPRTPA"
    ),
    # gyrA — Rv0006 — DNA gyrase subunit A (fluoroquinolone resistance)
    "gyrA": (
        "MSDLAREITPVNIEEELKSSYLDYAMSVIVGRALPDVRDGLKPVHRRVLYAMNVLGNDWNEL"
        "SHQDGKKIREVTDLVEFIEKQQHIRPHKAAIVLDRDSPGHKLMADLSTEPAPREALPNLLMQ"
        "ARLSEVTEDVFAQAREGERDIGDTVKAALVKELLTRLPGQIEDVRAITELIAQYLNELDPAVQ"
        "ALLERDIKGKASQLLKLQDQLIAQIAAELPAPQHLSGKIIDMLDDAPQFAAFLFQPDLKQQQ"
        "LMDEAIQALETALAQKAQATIPGELSGPELSTLLAAFATRQAQAAQALHAQRAQLQALRERLQ"
    ),
    # pncA — Rv2043c — pyrazinamidase (pyrazinamide resistance)
    "pncA": (
        "MRALIIVDQQNLLGQIEQQLASRGFSVVVASNEDSITLLKAMQEAIDQNNVDIVLLDHQNPQ"
        "QKGHPEWVDFAVPQAKAKGKVVFNHQEVNHHYDFFLQELADKLGFPVAHGFWKGCPVHIGA"
        "PHEVFYALQKIGVEKDRDAQFNQLIESIRQAIGLNMLCGDDAFVSAVCHPDFIDRISSQLPR"
        "VQ"
    ),
    # rpsL — Rv0682 — 30S ribosomal protein S12 (streptomycin resistance)
    "rpsL": (
        "MPVTNKSSRKKRAGKTRTKSAPKAQNAAAKPAAPRPKRAAAGKAGKTRRAKRKAPRAK"
        "RQTLTTGRGEGKKAPAKGTKPTRRAATPQRAATAKRRATTAAKPRAGKTKPK"
    ),
}

# ---------------------------------------------------------------------------
# WHO Catalog classification for known variants
# Grade:
#   1 = Confirmed resistance
#   2 = Associated with resistance
#   3 = Uncertain significance
#   4 = Not associated
# ---------------------------------------------------------------------------
WHO_CATALOG: dict[str, dict] = {
    # katG
    "katG_Ser315Thr": {"drug": "Isoniazid",     "grade": 1, "label": "Confirmed"},
    "katG_Ser315Asn": {"drug": "Isoniazid",     "grade": 1, "label": "Confirmed"},
    "katG_Ser315Ile": {"drug": "Isoniazid",     "grade": 2, "label": "Associated"},
    "katG_Ser315Arg": {"drug": "Isoniazid",     "grade": 2, "label": "Associated"},
    "katG_Arg463Leu": {"drug": "Isoniazid",     "grade": 3, "label": "Uncertain"},
    # rpoB — RRDR region (codons 426–452)
    "rpoB_Asp435Val": {"drug": "Rifampicin",    "grade": 1, "label": "Confirmed"},
    "rpoB_Asp435Tyr": {"drug": "Rifampicin",    "grade": 1, "label": "Confirmed"},
    "rpoB_His445Asp": {"drug": "Rifampicin",    "grade": 1, "label": "Confirmed"},
    "rpoB_His445Tyr": {"drug": "Rifampicin",    "grade": 1, "label": "Confirmed"},
    "rpoB_His445Asn": {"drug": "Rifampicin",    "grade": 2, "label": "Associated"},
    "rpoB_Leu430Pro": {"drug": "Rifampicin",    "grade": 1, "label": "Confirmed"},
    "rpoB_Ser450Leu": {"drug": "Rifampicin",    "grade": 1, "label": "Confirmed"},
    "rpoB_Ser450Trp": {"drug": "Rifampicin",    "grade": 1, "label": "Confirmed"},
    "rpoB_Ser450Phe": {"drug": "Rifampicin",    "grade": 2, "label": "Associated"},
    "rpoB_Asn437Asp": {"drug": "Rifampicin",    "grade": 1, "label": "Confirmed"},
    # inhA
    "inhA_Ser94Ala":  {"drug": "Isoniazid",     "grade": 1, "label": "Confirmed"},
    "inhA_Ile194Thr": {"drug": "Ethionamide",   "grade": 1, "label": "Confirmed"},
    # embB
    "embB_Met306Ile": {"drug": "Ethambutol",    "grade": 1, "label": "Confirmed"},
    "embB_Met306Val": {"drug": "Ethambutol",    "grade": 1, "label": "Confirmed"},
    "embB_Met306Leu": {"drug": "Ethambutol",    "grade": 2, "label": "Associated"},
    "embB_Gln497Arg": {"drug": "Ethambutol",    "grade": 2, "label": "Associated"},
    # gyrA
    "gyrA_Asp94Gly":  {"drug": "Fluoroquinolones", "grade": 1, "label": "Confirmed"},
    "gyrA_Asp94Asn":  {"drug": "Fluoroquinolones", "grade": 1, "label": "Confirmed"},
    "gyrA_Asp94Ala":  {"drug": "Fluoroquinolones", "grade": 1, "label": "Confirmed"},
    "gyrA_Ala90Val":  {"drug": "Fluoroquinolones", "grade": 1, "label": "Confirmed"},
    "gyrA_Ser91Pro":  {"drug": "Fluoroquinolones", "grade": 2, "label": "Associated"},
    # pncA
    "pncA_His57Asp":  {"drug": "Pyrazinamide",  "grade": 2, "label": "Associated"},
    "pncA_Trp68Gly":  {"drug": "Pyrazinamide",  "grade": 2, "label": "Associated"},
    # rpsL
    "rpsL_Lys43Arg":  {"drug": "Streptomycin",  "grade": 1, "label": "Confirmed"},
    "rpsL_Lys88Arg":  {"drug": "Streptomycin",  "grade": 1, "label": "Confirmed"},
}


# ---------------------------------------------------------------------------
# SECTION 2 — Module 1: Sequence Validation
# ---------------------------------------------------------------------------

def load_variant_dictionary(filepath: str = "variant_names.txt") -> set:
    """
    Load the set of variant feature names the models were trained on.

    Args:
        filepath: Path to a plain-text file with one variant per line.
                  e.g.  katG_Ser315Thr
                        rpoB_Ser450Leu

    Returns:
        A set of variant name strings.
    """
    try:
        with open(filepath, "r") as f:
            variants = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded {len(variants)} variants from '{filepath}'.")
        return variants
    except FileNotFoundError:
        logger.warning(f"'{filepath}' not found. Using WHO catalog as fallback dictionary.")
        return set(WHO_CATALOG.keys())


def parse_fasta(fasta_content: str) -> tuple[str, str]:
    """
    Parse raw FASTA text into (header, sequence).

    Handles:
      - Single-record FASTA
      - Multi-record FASTA (returns the first record; warns if >1)
      - Plain nucleotide strings (no header)

    Args:
        fasta_content: Raw string from file upload or text area.

    Returns:
        (header, sequence) where header may be empty string.

    Raises:
        ValueError: If the cleaned sequence is empty.
    """
    lines = fasta_content.strip().splitlines()
    header = ""
    seq_lines = []
    record_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            record_count += 1
            if record_count == 1:
                header = stripped[1:]   # drop the ">"
            elif record_count == 2:
                logger.warning("Multi-record FASTA detected. Only the first record will be used.")
        elif record_count <= 1:
            seq_lines.append(stripped)

    sequence = "".join(seq_lines).upper()

    # If no header was found, treat the whole input as a plain sequence
    if not sequence and not header:
        sequence = "".join(lines).upper()

    if not sequence:
        raise ValueError("No nucleotide sequence found in the provided input.")

    return header, sequence


def validate_sequence(sequence: str, gene_name: Optional[str] = None) -> dict:
    """
    Validate a nucleotide sequence for downstream processing.

    Checks:
      1. Only valid IUPAC DNA characters (ATGCNRYSWKMBDHV)
      2. Minimum length per gene (or 150 nt default)
      3. GC content within a plausible range for Mtb (~60–70%)

    Args:
        sequence:  Cleaned uppercase nucleotide string.
        gene_name: Optional gene name used for length thresholds.

    Returns:
        dict with keys:
          - valid (bool)
          - warnings (list[str])
          - errors (list[str])
          - gc_content (float)
          - length (int)
    """
    MIN_LENGTHS = {
        "katG":  2223,   # full gene; 741 codons
        "rpoB":   300,   # RRDR fragment acceptable
        "inhA":   840,
        "embB":  3300,
        "gyrA":   300,   # QRDR fragment acceptable
        "pncA":   561,
        "rpsL":   375,
    }
    VALID_CHARS = set("ATGCNRYSWKMBDHV")
    errors, warnings = [], []

    # 1. Character check
    invalid = set(sequence) - VALID_CHARS
    if invalid:
        errors.append(f"Invalid characters detected: {sorted(invalid)}. Only IUPAC DNA characters are allowed.")

    # 2. Length check
    min_len = MIN_LENGTHS.get(gene_name, 150) if gene_name else 150
    if len(sequence) < min_len:
        warnings.append(
            f"Sequence length ({len(sequence)} nt) is shorter than the expected minimum "
            f"for {gene_name or 'an unknown gene'} ({min_len} nt). Results may be incomplete."
        )

    # 3. Ambiguous base check
    n_count = sequence.count("N")
    if n_count > 0:
        pct = n_count / len(sequence) * 100
        if pct > 5:
            warnings.append(f"High ambiguity: {n_count} 'N' bases ({pct:.1f}%). Assembly quality may be low.")

    # 4. GC content (Mtb is ~65% GC)
    gc = (sequence.count("G") + sequence.count("C")) / max(len(sequence), 1) * 100
    if not (55 <= gc <= 75):
        warnings.append(
            f"GC content is {gc:.1f}%, outside the typical Mtb range (55–75%). "
            "Verify this is an Mtb sequence."
        )

    return {
        "valid":      len(errors) == 0,
        "errors":     errors,
        "warnings":   warnings,
        "gc_content": round(gc, 2),
        "length":     len(sequence),
    }


# ---------------------------------------------------------------------------
# SECTION 3 — Module 2: Variant Extraction
# ---------------------------------------------------------------------------

def _translate(dna: str) -> str:
    """
    Translate a DNA string to a protein string.
    Uses BioPython if available; falls back to built-in CODON_TABLE.
    Stops at first stop codon.
    """
    dna = dna.upper().replace("U", "T")   # handle RNA input

    if BIOPYTHON_AVAILABLE:
        protein = str(Seq(dna).translate(to_stop=True))
        return protein

    # Fallback: manual translation
    protein = []
    for i in range(0, len(dna) - 2, 3):
        codon = dna[i:i+3]
        aa = CODON_TABLE.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def _find_reading_frame(sequence: str, ref_protein: str, max_offset: int = 30) -> int:
    """
    Find the best reading frame offset (0–2) plus any start offset (0–max_offset)
    by maximising identity with the reference protein's first 30 AA.

    Returns the nucleotide offset to use before translating.
    Raises ValueError if no reasonable frame is found.
    """
    probe_len = 30   # compare first 30 AA of reference
    ref_probe = ref_protein[:probe_len]
    best_score = -1
    best_offset = 0

    for start in range(min(max_offset, len(sequence) // 3)):
        for frame in range(3):
            offset = start + frame
            if offset >= len(sequence):
                continue
            translated = _translate(sequence[offset:offset + probe_len * 3])
            score = sum(a == b for a, b in zip(translated, ref_probe))
            if score > best_score:
                best_score = score
                best_offset = offset

    if best_score < probe_len * 0.5:
        raise ValueError(
            f"Could not find a matching reading frame (best identity: {best_score}/{probe_len}). "
            "Check that the correct gene name is selected and the sequence is in the 5'→3' sense strand."
        )

    logger.info(f"Best reading frame offset: {best_offset} nt (identity {best_score}/{probe_len})")
    return best_offset


def extract_variants_from_sequence(
    gene_name: str,
    sequence: str,
    variant_dict: Optional[set] = None,
) -> list[dict]:
    """
    Core variant extraction engine.

    Steps:
      1. Look up the H37Rv reference protein for gene_name.
      2. Find the correct reading frame in the submitted sequence.
      3. Translate the submitted sequence to protein.
      4. Walk codon-by-codon; record any amino acid mismatch.
      5. Format as '<gene>_<RefAA><pos><AltAA>' (three-letter codes).
      6. Annotate each variant against WHO_CATALOG and variant_dict.

    Args:
        gene_name:    One of the keys in REFERENCE_PROTEINS (e.g. 'katG').
        sequence:     Cleaned uppercase DNA string.
        variant_dict: Optional set of model-known variant names for flagging.

    Returns:
        List of dicts, each with:
          - variant_name  (str)  e.g. 'katG_Ser315Thr'
          - position       (int)  codon position in reference
          - ref_aa         (str)  reference amino acid (3-letter)
          - alt_aa         (str)  alternate amino acid (3-letter)
          - drug           (str)  associated drug or 'Unknown'
          - grade          (int)  1–4 or None
          - label          (str)  'Confirmed'/'Associated'/'Uncertain'/'Not in model'/'Novel'
          - in_model       (bool) whether it's in variant_dict
    """
    ref_protein = REFERENCE_PROTEINS.get(gene_name)
    if ref_protein is None:
        raise ValueError(
            f"Gene '{gene_name}' not in reference database. "
            f"Supported genes: {sorted(REFERENCE_PROTEINS.keys())}"
        )

    # Find reading frame
    try:
        offset = _find_reading_frame(sequence, ref_protein)
    except ValueError as e:
        logger.error(str(e))
        raise

    # Translate
    query_protein = _translate(sequence[offset:])
    logger.info(f"Translated query protein length: {len(query_protein)} AA")

    variants = []
    compare_len = min(len(ref_protein), len(query_protein))

    for pos_0 in range(compare_len):          # zero-indexed
        ref_aa_1 = ref_protein[pos_0]
        qry_aa_1 = query_protein[pos_0]
        if ref_aa_1 == qry_aa_1:
            continue

        pos_1 = pos_0 + 1                     # 1-indexed (biological convention)
        ref_aa_3 = AA_1TO3.get(ref_aa_1, ref_aa_1)
        alt_aa_3 = AA_1TO3.get(qry_aa_1, qry_aa_1)
        var_name = f"{gene_name}_{ref_aa_3}{pos_1}{alt_aa_3}"

        who_info = WHO_CATALOG.get(var_name, {})
        in_model = (variant_dict is not None) and (var_name in variant_dict)

        if who_info:
            label = who_info["label"]
        elif in_model:
            label = "In model (unclassified)"
        else:
            label = "Novel"

        variants.append({
            "variant_name": var_name,
            "position":     pos_1,
            "ref_aa":       ref_aa_3,
            "alt_aa":       alt_aa_3,
            "drug":         who_info.get("drug", "Unknown"),
            "grade":        who_info.get("grade"),
            "label":        label,
            "in_model":     in_model,
        })

    logger.info(f"Detected {len(variants)} variant(s) in {gene_name}.")
    return variants


# ---------------------------------------------------------------------------
# SECTION 4 — Module 3: Feature Mapping
# ---------------------------------------------------------------------------

def get_model_input(
    detected_variants: list[dict],
    variant_names_list: list[str],
) -> list[int]:
    """
    Convert detected mutation dicts into the binary feature vector
    expected by XGBoost (or any sklearn-compatible) model.

    Args:
        detected_variants:   Output of extract_variants_from_sequence().
        variant_names_list:  Ordered list of all feature names the model knows
                             (preserves column order used at training time).

    Returns:
        Binary list of ints (1 = mutation present, 0 = absent),
        length == len(variant_names_list).
    """
    detected_set = {v["variant_name"] for v in detected_variants}
    return [1 if name in detected_set else 0 for name in variant_names_list]


# ---------------------------------------------------------------------------
# SECTION 5 — High-Level Pipeline Function
# ---------------------------------------------------------------------------

def process_fasta(
    fasta_content: str,
    gene_name: str,
    variant_names_list: Optional[list[str]] = None,
    variant_dict_path: str = "variant_names.txt",
) -> dict:
    """
    End-to-end pipeline: raw FASTA → model-ready feature vector.

    Args:
        fasta_content:      Raw string from Streamlit text_area or file uploader.
        gene_name:          Target gene (e.g. 'katG', 'rpoB').
        variant_names_list: Ordered feature list for the model. If None, the
                            function loads from variant_dict_path.
        variant_dict_path:  Path to variant_names.txt.

    Returns:
        dict with keys:
          - header         (str)       FASTA header line
          - sequence       (str)       cleaned nucleotide sequence
          - validation     (dict)      output of validate_sequence()
          - variants       (list[dict])output of extract_variants_from_sequence()
          - feature_vector (list[int]) binary vector for the model
          - summary_table  (list[dict])UI-ready rows for st.dataframe()
          - success        (bool)
          - error_message  (str | None)
    """
    result = {
        "header": "",
        "sequence": "",
        "validation": {},
        "variants": [],
        "feature_vector": [],
        "summary_table": [],
        "success": False,
        "error_message": None,
    }

    # Step 1 — Parse
    try:
        header, sequence = parse_fasta(fasta_content)
        result["header"] = header
        result["sequence"] = sequence
    except ValueError as e:
        result["error_message"] = f"FASTA parsing error: {e}"
        return result

    # Step 2 — Validate
    validation = validate_sequence(sequence, gene_name)
    result["validation"] = validation
    if not validation["valid"]:
        result["error_message"] = "Sequence validation failed: " + "; ".join(validation["errors"])
        return result

    # Step 3 — Load variant dictionary
    variant_dict = load_variant_dictionary(variant_dict_path)
    if variant_names_list is None:
        variant_names_list = sorted(variant_dict)  # deterministic fallback order

    # Step 4 — Extract variants
    try:
        variants = extract_variants_from_sequence(gene_name, sequence, variant_dict)
        result["variants"] = variants
    except ValueError as e:
        result["error_message"] = str(e)
        return result

    # Step 5 — Build feature vector
    result["feature_vector"] = get_model_input(variants, variant_names_list)

    # Step 6 — Build summary table for the Streamlit UI
    result["summary_table"] = [
        {
            "Variant":   v["variant_name"],
            "Position":  v["position"],
            "Ref AA":    v["ref_aa"],
            "Alt AA":    v["alt_aa"],
            "Drug":      v["drug"],
            "WHO Grade": v["grade"] if v["grade"] else "—",
            "Category":  v["label"],
            "In Model":  "✓" if v["in_model"] else "✗",
        }
        for v in variants
    ]

    result["success"] = True
    return result


# ---------------------------------------------------------------------------
# SECTION 6 — CLI / Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Minimal smoke test with a synthetic katG fragment
    # Real usage: replace with actual sequence
    DEMO_FASTA = """\
>Mtb_isolate_demo katG fragment
ATGACCGGCAGCGCGCCGCCGCACATCCGCGATCCGGCGGCGCAGCTGGCGGCGGAGGGC
TTCAACCTGGCCAACCCGGACAAGATCACCAACGCCTTCCTGGAACAAAACCACCACGAA
GACCGCTTCCCGGCCAACCGCGGCGCCGTCAGCATCGACGACAAGGCGCTGATGGCGGAG
CTGGAGCAGATCGTCGACACCGTCGCGTCGCTGGCGAAGGAGGATCCGCACGGCCTGAAC
CAGATCGCGTTCTACGACAAGTTCCCGCCGATGGGCCTGACCGCCATCGGCCAGCTGTCC
AGCTTCCTGGCGTCGCGCCTGACCCACGCGCAGGAGCCGATGTTCCGCGGCCTGTCCACC
TACTTCGAAACCCGCTGGACCCAGGAGATCCTGGCGGCGACGCTGCAGGGCAAGCAGCAC
GCGCAGCCGTACGTCGGCGACCTCATCGACGCGCTGCCGTCGCCGGGCCGCCACGGCCAG
CGCCAGCGCCCGAACATCGTGCCCACGTCGCGCCCGGACTCGGCGGGCTTCCTGCAGAAC
GGCGCGCTGACCGCGGCGCTGCGCAGCCGCAGCCCGACCGACGCGGCGTTCCGCGAGAGC
ATCCGCCAGGGCCTGGCGGAGTTCAGCGGCCTGCGCAACGAGGCGATCCAGGCGTTCGGC
AAGGCGCAGCAGGACCTGGACCCGTTCCGCCGCGCGCAGATCCAGCACTTCGGCACGCTG
GGCAAGGTCGCGATGCACCGCGAGAGCTGGCTGCACCAGCTGCTGTTCGGCGGCCTGCAC
GAGCTGATCGACTTCCGCGACCTGCGCGGCCAGCGCATCGACATCGAGGAGCTGCTGGTG
CTGGTCGCGGGCAAGATCCGCGACCTGAAGCGCCAGATGGACGAGGGTGACCACGCGGTG
CAGTACGTGCCGCAGGACCGGCTGCCGCACGCCGTGCTGGCGCGCCTGCAGCAGCTGGCG
"""

    print("=" * 60)
    print("processor.py — Demo Run")
    print("=" * 60)

    output = process_fasta(
        fasta_content=DEMO_FASTA,
        gene_name="katG",
    )

    if output["success"]:
        print(f"\nHeader   : {output['header']}")
        print(f"Length   : {output['validation']['length']} nt")
        print(f"GC       : {output['validation']['gc_content']}%")
        print(f"Warnings : {output['validation']['warnings'] or 'None'}")
        print(f"\nVariants detected: {len(output['variants'])}")
        for row in output["summary_table"]:
            print(f"  {row['Variant']:30s}  Drug: {row['Drug']:20s}  Category: {row['Category']}")
        print(f"\nFeature vector (first 10): {output['feature_vector'][:10]}")
    else:
        print(f"\nERROR: {output['error_message']}")