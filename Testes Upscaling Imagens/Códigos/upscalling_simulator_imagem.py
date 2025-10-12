from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont

def criar_imagem_comparativa(original, baixa_res, simples, aprimorada, nome_arquivo_saida):
    """
    Cria uma única imagem de comparação lado a lado com 4 painéis e janelas de zoom.
    """
    print("Criando imagem de comparação final com 4 painéis...")

    area_zoom = (largura_original // 2, altura_original // 2, 
                 largura_original // 2 + 250, altura_original // 2 + 250)
    zoom_size = (250 * 2, 250 * 2)

    largura_thumb, altura_thumb = original.width // 2.5, original.height // 2.5
    largura_thumb, altura_thumb = int(largura_thumb), int(altura_thumb)
    
    thumb_original = original.resize((largura_thumb, altura_thumb), Image.Resampling.LANCZOS)
    thumb_baixa_res = baixa_res.resize((largura_thumb, altura_thumb), Image.Resampling.NEAREST)
    thumb_simples = simples.resize((largura_thumb, altura_thumb), Image.Resampling.LANCZOS)
    thumb_aprimorada = aprimorada.resize((largura_thumb, altura_thumb), Image.Resampling.LANCZOS)

    zoom_original = original.crop(area_zoom).resize(zoom_size, Image.Resampling.NEAREST)
    zoom_baixa_res = baixa_res.resize(original.size, Image.Resampling.NEAREST).crop(area_zoom).resize(zoom_size, Image.Resampling.NEAREST)
    zoom_simples = simples.crop(area_zoom).resize(zoom_size, Image.Resampling.NEAREST)
    zoom_aprimorada = aprimorada.crop(area_zoom).resize(zoom_size, Image.Resampling.NEAREST)
    
    padding = 50
    largura_total = (largura_thumb * 4) + (padding * 5)
    altura_total = altura_thumb + zoom_original.height + (padding * 4)
    canvas = Image.new('RGB', (largura_total, altura_total), 'white')
    draw = ImageDraw.Draw(canvas)

    try:
        fonte_titulo = ImageFont.truetype("arialbd.ttf", 40)
        fonte_legenda = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        fonte_titulo = ImageFont.load_default()
        fonte_legenda = ImageFont.load_default()

    pos_x1 = padding
    pos_x2 = largura_thumb + 2*padding
    pos_x3 = largura_thumb*2 + 3*padding
    pos_x4 = largura_thumb*3 + 4*padding
    pos_y_thumb = padding * 2
    pos_y_zoom = pos_y_thumb + altura_thumb + padding

    canvas.paste(thumb_original, (pos_x1, pos_y_thumb))
    canvas.paste(thumb_baixa_res, (pos_x2, pos_y_thumb))
    canvas.paste(thumb_simples, (pos_x3, pos_y_thumb))
    canvas.paste(thumb_aprimorada, (pos_x4, pos_y_thumb))
    
    canvas.paste(zoom_original, (pos_x1, pos_y_zoom))
    canvas.paste(zoom_baixa_res, (pos_x2, pos_y_zoom))
    canvas.paste(zoom_simples, (pos_x3, pos_y_zoom))
    canvas.paste(zoom_aprimorada, (pos_x4, pos_y_zoom))

    draw.text((largura_total/2, padding//2), "Comparativo de Técnicas de Upscaling", font=fonte_titulo, fill='black', anchor='mt')
    
    draw.text((pos_x1 + largura_thumb/2, pos_y_thumb - 10), "1. Original (Nativo)", font=fonte_legenda, fill='black', anchor='ms')
    draw.text((pos_x2 + largura_thumb/2, pos_y_thumb - 10), "2. Baixa Resolução", font=fonte_legenda, fill='black', anchor='ms')
    draw.text((pos_x3 + largura_thumb/2, pos_y_thumb - 10), "3. Upscaling Simples", font=fonte_legenda, fill='black', anchor='ms')
    draw.text((pos_x4 + largura_thumb/2, pos_y_thumb - 10), "4. Upscaling Aprimorado", font=fonte_legenda, fill='black', anchor='ms')

    for pos_x in [pos_x1, pos_x2, pos_x3, pos_x4]:
        draw.rectangle([pos_x, pos_y_zoom, pos_x + zoom_original.width, pos_y_zoom + zoom_original.height], outline='red', width=3)

    canvas.save(nome_arquivo_saida)
    print(f"✅ Imagem de comparação final (4 painéis) salva como '{nome_arquivo_saida}'")


try:
    imagem_original = Image.open("imagem_alta_resolucao.jpg").convert("RGB")
except FileNotFoundError:
    print("Erro: Coloque uma imagem chamada 'imagem_alta_resolucao.jpg' na pasta.")
    exit()

largura_original, altura_original = imagem_original.size
print(f"Resolução Nativa: {largura_original}x{altura_original}")

fator_escala = 0.5
nova_largura = int(largura_original * fator_escala)
nova_altura = int(altura_original * fator_escala)
imagem_baixa_res = imagem_original.resize((nova_largura, nova_altura), Image.Resampling.LANCZOS)
print(f"Renderização em baixa resolução simulada: {nova_largura}x{nova_altura}")

imagem_upscaled_simples = imagem_baixa_res.resize((largura_original, altura_original), Image.Resampling.BILINEAR)

imagem_upscaled_suave = imagem_baixa_res.resize((largura_original, altura_original), Image.Resampling.LANCZOS)
mascara = imagem_upscaled_suave.convert('L').filter(ImageFilter.FIND_EDGES)
mascara = ImageEnhance.Contrast(mascara).enhance(10.0)
mascara = mascara.filter(ImageFilter.GaussianBlur(radius=1))
imagem_com_nitidez_forte = imagem_upscaled_suave.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=4))
imagem_final_aprimorada = Image.composite(imagem_com_nitidez_forte, imagem_upscaled_suave, mascara)
enhancer = ImageEnhance.Contrast(imagem_final_aprimorada)
imagem_final_aprimorada = enhancer.enhance(1.1)

criar_imagem_comparativa(
    original=imagem_original,
    baixa_res=imagem_baixa_res, 
    simples=imagem_upscaled_simples,
    aprimorada=imagem_final_aprimorada,
    nome_arquivo_saida="comparacao_final_upscaling.png"
)