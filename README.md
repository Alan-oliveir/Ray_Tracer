RAY TRACER PARA ESFERAS E TORUS - CPS751
==========================================

Trabalho final da disciplina computação gráfica: sistema de ray tracing para esferas e torus.

FUNCIONALIDADES IMPLEMENTADAS:
-----------------------------
- Ray casting básico
- Intersecção com esferas
- Intersecção com torus
- Modelo de iluminação Phong
- Sombras (shadow feelers)
- Reflexões
- Transparência/refração

EFEITOS VISUAIS:
----------------
- Iluminação Phong - Componentes ambiente, difusa e especular
- Sombras - Shadow feelers para todas as fontes de luz
- Reflexões - Raios refletidos recursivos
- Transparência/Refração - Lei de Snell implementada
- Múltiplas luzes - Suporte a várias fontes de luz

ESTRUTURA DO CÓDIGO:
--------------------
- Material - Define propriedades dos materiais
- Light - Fontes de luz pontuais
- Sphere - Implementa esferas com intersecção
- Torus - Implementa torus com intersecção complexa
- RayTracer - Engine principal de ray tracing

COMO USAR:
----------
1. Instale a dependência: `pip install pillow numpy`
2. Execute o programa: `python raytracer.py`
3. O programa irá gerar duas imagens:
   - 'esfera_scene.png': Cena com esferas
   - 'torus_scene.png': Cena com torus e esferas
4. As imagens serão salvas no diretório atual

PARÂMETROS CONFIGURÁVEIS:
------------------------
- Resolução da imagem (WIDTH, HEIGHT)
- Posição da câmera (CAMERA_POS)
- Posição e propriedades das luzes
- Materiais dos objetos (cor, reflexão, transparência)

CARACTERÍSTICAS TÉCNICAS:
-------------------------
- Resolução: 800x600 pixels (configurável)
- Profundidade de recursão: 5 níveis
- Algoritmo: Ray tracing backward (da câmera para a cena)
- Intersecção com torus: Resolve equação quártica usando método de Ferrari
- Anti-aliasing: Pode ser adicionado facilmente

---
**Baseado no material da Aula 14 - Ray Tracing**  
**Disciplina**: CPS751 - Computação Gráfica  
**Observação**: O código foi desenvolvido em Python para fins acadêmicos, utilizando bibliotecas como Pillow e NumPy 
para manipulação de imagens e cálculos matemáticos. A implementação é didática e visa demonstrar os conceitos de ray 
tracing, interseção de objetos e iluminação. Para aplicacações mais complexas, recomenda-se o uso de linguagens de baixo
nível como C++ ou Rust para maior performance.

---
## Aluno

**Alan Gonçalves**  
*Engenharia Eletrônica e de Computação - UFRJ*  
*Disciplina: Computação Gráfica*  
*Professor: Ricardo Farias*  
*Período: 2025.1*