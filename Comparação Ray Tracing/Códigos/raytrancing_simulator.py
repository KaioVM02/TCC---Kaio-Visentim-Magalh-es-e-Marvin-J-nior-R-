import numpy as np
from PIL import Image, ImageDraw, ImageFont

largura, altura = 800, 600
camera = np.array([0, 0.5, 2.5])
luz = {'position': np.array([5, 5, 5]), 'intensity': 1.5, 'color': np.array([255, 255, 255])}
cor_fundo = np.array([100, 149, 237]) # Cor de céu
esferas = [
    {'center': np.array([0, 0, -1]), 'radius': 0.5, 'color': np.array([255, 0, 0]), 'reflectivity': 0.3, 'shininess': 100},
    {'center': np.array([-1.2, 0.2, -1.5]), 'radius': 0.7, 'color': np.array([0, 150, 255]), 'reflectivity': 0.8, 'shininess': 500},
    {'center': np.array([1.0, -0.3, -1]), 'radius': 0.2, 'color': np.array([0, 255, 0]), 'reflectivity': 0.1, 'shininess': 10}
]
plano_chao = {'point': np.array([0, -0.5, 0]), 'normal': np.array([0, 1, 0]), 'color': np.array([200, 200, 200]), 'reflectivity': 0.4, 'shininess': 50}
objetos = esferas + [plano_chao]

def normalizar(vetor):
    norm = np.linalg.norm(vetor)
    if norm == 0: return vetor
    return vetor / norm

def intersecao_esfera(o, d, s):
    oc = o - s['center']
    a, b, c = np.dot(d, d), 2 * np.dot(oc, d), np.dot(oc, oc) - s['radius']**2
    disc = b**2 - 4*a*c
    if disc < 0: return None
    sqrt_disc = np.sqrt(disc)
    t1, t2 = (-b - sqrt_disc) / (2*a), (-b + sqrt_disc) / (2*a)
    if t1 > 1e-4: return t1
    if t2 > 1e-4: return t2
    return None

def intersecao_plano(o, d, p):
    denom = np.dot(d, p['normal'])
    if abs(denom) > 1e-6:
        t = np.dot(p['point'] - o, p['normal']) / denom
        if t > 1e-4: return t
    return None

def trace_ray_componentes(origem, direcao, depth):
    if depth <= 0:
        return {'base': cor_fundo, 'shading_sombras': cor_fundo, 'reflexos': cor_fundo, 'final': cor_fundo}

    dist, obj_atingido = float('inf'), None
    for obj in objetos:
        d = (intersecao_esfera if 'radius' in obj else intersecao_plano)(origem, direcao, obj)
        if d and d < dist: dist, obj_atingido = d, obj
    
    if not obj_atingido:
        return {'base': cor_fundo, 'shading_sombras': cor_fundo, 'reflexos': cor_fundo, 'final': cor_fundo}

    ponto = origem + direcao * dist
    normal = normalizar(ponto - obj_atingido['center']) if 'radius' in obj_atingido else obj_atingido['normal']
    cor_base = obj_atingido['color']
    cor_ambiente = cor_base * 0.1
    cor_difusa, cor_especular = np.array([0,0,0]), np.array([0,0,0])
    
    em_sombra = False
    direcao_luz = normalizar(luz['position'] - ponto)
    ponto_inicio_sombra = ponto + normal * 1e-4
    for outro_obj in objetos:
        if outro_obj is not obj_atingido:
            if (intersecao_esfera if 'radius' in outro_obj else intersecao_plano)(ponto_inicio_sombra, direcao_luz, outro_obj):
                em_sombra = True
                break
    
    if not em_sombra:
        intensidade_difusa = max(0, np.dot(normal, direcao_luz))
        cor_difusa = cor_base * intensidade_difusa * luz['intensity']
        dir_reflexo_luz = 2 * np.dot(normal, direcao_luz) * normal - direcao_luz
        intensidade_especular = max(0, np.dot(dir_reflexo_luz, -direcao))**obj_atingido['shininess']
        cor_especular = luz['color'] * intensidade_especular
        
    cor_com_shading = cor_ambiente + cor_difusa + cor_especular

    fator_reflexo = obj_atingido['reflectivity']
    cor_refletida = cor_fundo
    if fator_reflexo > 0:
        dir_reflexo = direcao - 2 * np.dot(direcao, normal) * normal
        cor_refletida = trace_ray_componentes(ponto, dir_reflexo, depth - 1)['final']
    
    cor_com_reflexos = cor_base * (1 - fator_reflexo) + cor_refletida * fator_reflexo
    cor_final = cor_com_shading * (1 - fator_reflexo) + cor_refletida * fator_reflexo
    
    return {'base': np.clip(cor_base,0,255), 'shading_sombras': np.clip(cor_com_shading,0,255),
            'reflexos': np.clip(cor_com_reflexos,0,255), 'final': np.clip(cor_final,0,255)}

def renderizar_raster_fake(y, x):
    px, py = (x + 0.5) / largura * 2 - 1, -(y + 0.5) / altura * 2 + 1
    px *= largura / altura
    direcao = normalizar(np.array([px, py, -1]))
    dist, obj_atingido = float('inf'), None
    for obj in objetos:
        d = (intersecao_esfera if 'radius' in obj else intersecao_plano)(camera, direcao, obj)
        if d and d < dist: dist, obj_atingido = d, obj
    if not obj_atingido:
        return {'base': cor_fundo, 'shading_sombras': cor_fundo, 'reflexos': cor_fundo, 'final': cor_fundo}
    
    ponto = camera + direcao * dist
    normal = normalizar(ponto - obj_atingido['center']) if 'radius' in obj_atingido else obj_atingido['normal']
    cor_base, cor_ambiente = obj_atingido['color'], obj_atingido['color'] * 0.1
    direcao_luz = normalizar(luz['position'] - ponto)
    intensidade_difusa = max(0, np.dot(normal, direcao_luz))
    cor_difusa = cor_base * intensidade_difusa * luz['intensity']
    dir_reflexo_luz = 2 * np.dot(normal, direcao_luz) * normal - direcao_luz
    intensidade_especular = max(0, np.dot(dir_reflexo_luz, -direcao))**obj_atingido['shininess']
    cor_especular = luz['color'] * intensidade_especular
    cor_iluminada = cor_ambiente + cor_difusa + cor_especular

    cor_com_sombra = cor_iluminada
    if obj_atingido is plano_chao:
        for esfera in esferas:
            dist_chao = (esfera['center'][1] - plano_chao['point'][1])
            proj_x = esfera['center'][0] - (luz['position'][0] * dist_chao / luz['position'][1])
            proj_z = esfera['center'][2] - (luz['position'][2] * dist_chao / luz['position'][1])
            if (ponto[0] - proj_x)**2 / (esfera['radius']**2) + (ponto[2] - proj_z)**2 / (esfera['radius']**2) < 1:
                cor_com_sombra = cor_ambiente
                break
                
    def sample_env_map(direction):
        return np.array([200, 200, 200]) * (1 - direction[1]) + cor_fundo * direction[1]
    
    fator_reflexo = obj_atingido['reflectivity']
    cor_refletida_falsa = cor_fundo
    if fator_reflexo > 0:
        dir_reflexo = direcao - 2 * np.dot(direcao, normal) * normal
        cor_refletida_falsa = sample_env_map(dir_reflexo)
        
    cor_com_reflexos = cor_base * (1 - fator_reflexo) + cor_refletida_falsa * fator_reflexo
    cor_final = cor_com_sombra * (1 - fator_reflexo) + cor_refletida_falsa * fator_reflexo

    return {'base': np.clip(cor_base,0,255), 'shading_sombras': np.clip(cor_com_sombra,0,255),
            'reflexos': np.clip(cor_com_reflexos,0,255), 'final': np.clip(cor_final,0,255)}

def criar_imagem_comparativa(prefixo, titulos):
    imagens = [Image.open(f'{prefixo}_{i+1}.png') for i in range(4)]
    largura_img, altura_img = imagens[0].size
    padding = 40
    canvas = Image.new('RGB', (largura_img * 2 + padding * 3, altura_img * 2 + padding * 3), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        fonte = ImageFont.truetype("arialbd.ttf", 30)
    except IOError:
        fonte = ImageFont.load_default()
    posicoes = [(padding, padding), (largura_img + 2 * padding, padding),
                (padding, altura_img + 2 * padding), (largura_img + 2 * padding, altura_img + 2 * padding)]
    for img, pos, titulo in zip(imagens, posicoes, titulos):
        canvas.paste(img, pos)
        draw.text((pos[0] + largura_img / 2, pos[1] + altura_img + 15), titulo, font=fonte, fill='black', anchor='ma')
    canvas.save(f'{prefixo}_comparativo.png')

print("Iniciando renderização com Ray Tracing Real...")
rt_imagens = [np.zeros((altura, largura, 3), dtype=np.uint8) for _ in range(4)]
for y in range(altura):
    for x in range(largura): 
        px, py = (x + 0.5) / largura * 2 - 1, -(y + 0.5) / altura * 2 + 1
        px *= largura / altura
        direcao = normalizar(np.array([px, py, -1]))
        componentes = trace_ray_componentes(camera, direcao, depth=3)
        rt_imagens[0][y, x] = componentes['base']
        rt_imagens[1][y, x] = componentes['shading_sombras']
        rt_imagens[2][y, x] = componentes['reflexos']
        rt_imagens[3][y, x] = componentes['final']

print("Salvando imagens intermediárias do Ray Tracing...")
for i in range(4):
    Image.fromarray(rt_imagens[i], 'RGB').save(f'rt_{i+1}.png')

titulos_rt = ["1. Cor Base", "2. Shading + Sombras", "3. Reflexos", "4. Imagem Final (RT)"]
criar_imagem_comparativa('rt', titulos_rt)
print("✅ Imagem 'rt_comparativo.png' gerada.")

print("\nIniciando renderização com técnicas 'Falsas' (Rasterização)...")
raster_imagens = [np.zeros((altura, largura, 3), dtype=np.uint8) for _ in range(4)]
for y in range(altura):
    for x in range(largura):
        componentes = renderizar_raster_fake(y, x)
        raster_imagens[0][y, x] = componentes['base']
        raster_imagens[1][y, x] = componentes['shading_sombras']
        raster_imagens[2][y, x] = componentes['reflexos']
        raster_imagens[3][y, x] = componentes['final']

print("Salvando imagens intermediárias da renderização falsa...")
for i in range(4):
    Image.fromarray(raster_imagens[i], 'RGB').save(f'raster_{i+1}.png')

titulos_raster = ["1. Cor Base", "2. Shading + Sombras Falsas", "3. Reflexos Falsos", "4. Imagem Final (Falsa)"]
criar_imagem_comparativa('raster', titulos_raster)
print("✅ Imagem 'raster_comparativo.png' gerada.")

print("\nCriando a comparação final Lado a Lado...")
img_rt = Image.open('rt_4.png')
img_raster = Image.open('raster_4.png')
canvas_final = Image.new('RGB', (largura * 2 + 150, altura + 100), 'white')
draw_final = ImageDraw.Draw(canvas_final)
canvas_final.paste(img_raster, (50, 50))
canvas_final.paste(img_rt, (largura + 100, 50))
try:
    fonte_final = ImageFont.truetype("arialbd.ttf", 30)
except IOError:
    fonte_final = ImageFont.load_default()
draw_final.text((50 + largura/2, 20), "Renderização Falsa (Rasterização)", font=fonte_final, fill='black', anchor='mt')
draw_final.text((100 + largura + largura/2, 20), "Renderização com Ray Tracing", font=fonte_final, fill='black', anchor='mt')
canvas_final.save('comparacao_final_RT_vs_Raster.png')

print("\nProcesso completo!")
print("✅ Imagem de comparação final salva como 'comparacao_final_RT_vs_Raster.png'")