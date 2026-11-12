# pip install -U transformers torch

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def topk_for_all_masks(
    text: str,
    model_name: str = "distilbert-base-uncased",
    top_k: int = 10,
    device: str | None = None,
):
    """
    BERT(MLM) 계열(여기서는 DistilBERT)로 문장 안의 [MASK] 위치들에 대해
    각 마스크별 top-k 후보 토큰/확률을 출력합니다.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    # DistilBERT는 [MASK] 토큰을 사용
    mask_id = tokenizer.mask_token_id
    assert tokenizer.mask_token is not None, "This tokenizer has no mask token."

    # tokenize
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # mask 위치 찾기
    mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=False).flatten().tolist()
    if len(mask_positions) == 0:
        raise ValueError(f"No mask token found. Use '{tokenizer.mask_token}' in the text.")

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    results = []
    for pos in mask_positions:
        # 해당 위치 vocab logits -> 확률
        probs = torch.softmax(logits[0, pos], dim=-1)  # [vocab_size]
        top_probs, top_ids = torch.topk(probs, k=top_k)

        # 사람이 읽을 수 있게 변환
        candidates = []
        for p, tid in zip(top_probs.tolist(), top_ids.tolist()):
            tok = tokenizer.convert_ids_to_tokens(tid)
            # subword('##')를 보기 좋게
            tok_clean = tok.replace("##", "")
            candidates.append({"token": tok, "token_clean": tok_clean, "prob": p})
        results.append({"mask_position": pos, "candidates": candidates})

    return results

if __name__ == "__main__":
    # 예시 1) single mask
    text1 = "a [MASK] is reading a book."
    out1 = topk_for_all_masks(text1, top_k=8)
    print("INPUT:", text1)
    for r in out1:
        print(f"\n[MASK] position={r['mask_position']}")
        for c in r["candidates"]:
            print(f"  {c['token']:<12}  prob={c['prob']:.4f}")

    # 예시 2) multiple masks (한 번의 forward로 각 mask 위치별 분포를 동시에 출력)
    text2 = "a [MASK] is [MASK] a book."
    out2 = topk_for_all_masks(text2, top_k=8)
    print("\n\nINPUT:", text2)
    for r in out2:
        print(f"\n[MASK] position={r['mask_position']}")
        for c in r["candidates"]:
            print(f"  {c['token']:<12}  prob={c['prob']:.4f}")
