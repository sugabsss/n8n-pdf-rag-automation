from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pdf2image import convert_from_path, pdfinfo_from_path
from pathlib import Path
import pytesseract
from PIL import Image, ImageOps
import cv2
import numpy as np # Importe o NumPy
import os
import gc # Garbage Collector para forçar a liberação de memória

app = FastAPI()
router = APIRouter()

# --- Constantes e Configurações ---
CAMINHO_DO_SCRIPT = Path(__file__).parent
DIRETORIO_PAI = CAMINHO_DO_SCRIPT.parent
PASTA_PDFS = DIRETORIO_PAI / "arquivos_pdf"
PASTA_IMAGENS = DIRETORIO_PAI / "images"
PASTA_TEXTO_IMAGEM = DIRETORIO_PAI / "texto_imagem"

# Garanta que as pastas de saída existam
os.makedirs(PASTA_IMAGENS, exist_ok=True)
os.makedirs(PASTA_TEXTO_IMAGEM, exist_ok=True)

# Verifique se o Tesseract está instalado corretamente
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# --- Funções de Processamento de Imagem ---

def preprocess_image(pil_img):
    """Aplica pré-processamento básico na imagem para melhorar o OCR."""
    img = pil_img.convert("L") # Converte para escala de cinza
    # Inverter a imagem pode ajudar em alguns casos, mas teste para ver se melhora no seu caso
    # img = ImageOps.invert(img) 
    img = img.point(lambda x: 0 if x < 180 else 255, '1')
    return img

def detectar_blocos(pil_img):
    """Usa OpenCV para detectar áreas de texto/tabela diretamente de uma imagem PIL."""
    # Converte a imagem PIL para um array NumPy que o OpenCV entende
    # Evita salvar e ler do disco desnecessariamente
    img_cv = np.array(pil_img.convert('RGB'))
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
    _, thresh = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocos = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 40 and h > 15:  # Filtro de ruído ajustado
            blocos.append((x, y, w, h))
    # Ordena os blocos de cima para baixo
    return sorted(blocos, key=lambda b: b[1])

def processar_pdf_em_background(caminho_pdf_str: str, nome_base: str):
    """
    Função que executa o trabalho pesado de OCR. 
    Projetada para ser chamada por uma Background Task no FastAPI.
    """
    try:
        # 1. Obter o número total de páginas sem carregar o PDF
        info = pdfinfo_from_path(caminho_pdf_str)
        total_paginas = info['Pages']

        # 2. Iterar sobre cada página, uma de cada vez
        for i in range(1, total_paginas + 1):
            print(f"Processando página {i}/{total_paginas} do arquivo {nome_base}...")
            
            # 3. Converter APENAS a página atual, com um DPI mais razoável
            imagem_pagina = convert_from_path(
                caminho_pdf_str,
                dpi=300,  # DPI reduzido para 300 (ótimo para OCR e muito mais leve)
                first_page=i,
                last_page=i
            )

            if not imagem_pagina:
                continue

            imagem = imagem_pagina[0]

            # Salvar a imagem da página (opcional, mas útil para debug)
            nome_imagem_saida = f"{nome_base}_pagina_{i}.png"
            caminho_imagem_saida = PASTA_IMAGENS / nome_imagem_saida
            imagem.save(caminho_imagem_saida, "PNG")

            # Processamento OCR da página
            blocos = detectar_blocos(imagem)
            texto_final = []

            for (x, y, w, h) in blocos:
                recorte = imagem.crop((x, y, x + w, y + h))
                
                # A função de pré-processamento pode ser aplicada aqui se necessário
                # recorte_processado = preprocess_image(recorte)

                # Escolha do PSM (Page Segmentation Mode) pode ser simplificada
                config = r'--oem 3 --psm 4 -l por' # PSM 4 é um bom padrão para blocos

                texto = pytesseract.image_to_string(recorte, config=config)
                texto_final.append(texto.strip())

            # Salvar o texto extraído da página
            caminho_texto_saida = PASTA_TEXTO_IMAGEM / f"{nome_base}_pagina_{i}.txt"
            with open(caminho_texto_saida, "w", encoding="utf-8") as f:
                f.write("\n\n".join(texto_final))

            # 4. Liberar a memória explicitamente
            del imagem
            del imagem_pagina
            gc.collect() # Chama o coletor de lixo

        print(f"Processamento do arquivo {nome_base} concluído com sucesso.")

    except Exception as e:
        # Em um sistema real, você logaria este erro em um arquivo ou sistema de logs
        print(f"Erro ao processar o arquivo {nome_base}: {str(e)}")


@router.post("/converter-pdf/{nome_arquivo}")
async def converter_pdf_por_nome(nome_arquivo: str, background_tasks: BackgroundTasks):
    caminho_pdf = PASTA_PDFS / nome_arquivo
    if not caminho_pdf.is_file():
        raise HTTPException(status_code=404, detail=f"O arquivo '{nome_arquivo}' não foi encontrado.")

    nome_base = caminho_pdf.stem
    
    # Adiciona a tarefa pesada para ser executada em segundo plano
    background_tasks.add_task(processar_pdf_em_background, str(caminho_pdf), nome_base)
    
    # Retorna uma resposta IMEDIATA para o usuário
    return JSONResponse(
        status_code=202, # 202 Accepted
        content={"mensagem": f"O processamento do arquivo '{nome_arquivo}' foi iniciado. O resultado será salvo no servidor."}
    )


@router.get("/images/{nome_imagem}")
async def get_image(nome_imagem: str):
    caminho_imagem = PASTA_IMAGENS / nome_imagem
    if not caminho_imagem.is_file():
        raise HTTPException(status_code=404, detail="Imagem não encontrada.")
    return FileResponse(caminho_imagem)

app.include_router(router)