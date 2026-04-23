def build_final_prompt(
    doctor_prompt: str,
    pdf_chunks: list[dict],
    med_chunks: list[dict],
) -> str:
    lines = []

    # System instruction
    lines.append("=== SYSTEM INSTRUCTION ===")
    if doctor_prompt.strip():
        lines.append(doctor_prompt.strip())
    else:
        lines.append(
            "You are a Senior Clinical Assistant. Analyze the provided context and answer clinically. "
            "Perform inference — do not only look for exact keywords. "
            "Identify risks, contradictions, and patient-specific concerns. "
            "Cite [PDF-N] for patient document findings and [MED-N] for medical literature."
        )
    lines.append("")

    # Patient document context
    lines.append("=== PATIENT DOCUMENT CONTEXT [PDF] ===")
    if pdf_chunks:
        for i, c in enumerate(pdf_chunks, 1):
            lines.append(f"[PDF-{i}]")
            lines.append(c["text"])
            lines.append("")
    else:
        lines.append("No patient document uploaded.\n")

    # Medical literature context
    lines.append("=== MEDICAL LITERATURE CONTEXT [MED] ===")
    if med_chunks:
        for i, c in enumerate(med_chunks, 1):
            lines.append(f"[MED-{i}] Source: {c.get('source', 'Medical KB')}")
            lines.append(c["text"])
            lines.append("")
    else:
        lines.append("No medical literature available.\n")

    # Rules
    lines.append("=== SYNTHESIS RULES ===")
    lines.append(
        "1. [MED] sources take priority for clinical guidelines and drug safety.\n"
        "2. [PDF] sources provide patient-specific context and history.\n"
        "3. If patient history (PDF) shows an allergy or contraindication, flag it explicitly.\n"
        "4. Structure answer as: Assessment → Evidence → Recommendation.\n"
        "5. Always cite inline with [PDF-N] or [MED-N]."
    )

    return "\n".join(lines)