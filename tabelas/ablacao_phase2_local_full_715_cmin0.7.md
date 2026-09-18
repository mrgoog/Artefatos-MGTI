<!-- gerado por scripts/gerar_tabelas_ablacao.py — não editar à mão -->
# Ablação de arquitetura — phase2_local_full_715, C_min=0.7, n=715

| variante | descricao | acuracia_cob1 | auroc | aurc | risco_cob03 | risco_cob05 | delta_aurc_vs_a0 | ic_lo | ic_hi | significativo |
|---|---|---|---|---|---|---|---|---|---|---|
| A0 | atual: w=alpha, argmax(w*c_calib) | 0.259 | 0.693 | 0.636 | 0.605 | 0.606 | 0.0 | nan | nan | False |
| A2 | w=alpha sem recusas | 0.259 | 0.696 | 0.632 | 0.6 | 0.609 | -0.004 | -0.017 | 0.01 | False |
| A3 | w=informatividade (AUROC-0.5) | 0.271 | 0.693 | 0.611 | 0.586 | 0.602 | -0.025 | -0.061 | 0.014 | False |
| A1 | sem w: argmax(c_calib) | 0.266 | 0.696 | 0.622 | 0.59 | 0.604 | -0.014 | -0.03 | 0.001 | False |
| A4 | argmax(c_calib) + score=media(c_calib) | 0.266 | 0.722 | 0.59 | 0.521 | 0.605 | -0.046 | -0.068 | -0.023 | True |
| A5 | argmax(c_bruta) + score=media(c_bruta) | 0.27 | 0.738 | 0.568 | 0.49 | 0.596 | -0.068 | -0.116 | -0.018 | True |
| A7 | sem w: argmax(c_bruta) + score=max(c_bruta) | 0.27 | 0.701 | 0.623 | 0.578 | 0.583 | -0.013 | -0.053 | 0.028 | False |
| A6 | logistica(media,min,max,std,n_recusas) | 0.266 | 0.712 | 0.6 | 0.545 | 0.602 | -0.036 | -0.058 | -0.015 | True |
| B0 | sempre gemma4_e4b + score=c_bruta dele | 0.248 | 0.749 | 0.627 | 0.574 | 0.591 | -0.009 | -0.049 | 0.032 | False |
