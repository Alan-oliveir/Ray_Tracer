"""
RAY TRACER PARA ESFERAS E TORUS - CPS751
==========================================

COMO USAR:
----------
1. Execute o programa: python raytracer.py
2. O programa irá gerar duas imagens:
   - 'esfera_scene.png': Cena com esferas
   - 'torus_scene.png': Cena com torus e esferas
3. As imagens serão salvas no diretório atual

PARÂMETROS CONFIGURÁVEIS:
------------------------
- Resolução da imagem (WIDTH, HEIGHT)
- Posição da câmera (CAMERA_POS)
- Posição e propriedades das luzes
- Materiais dos objetos (cor, reflexão, transparência)

FUNCIONALIDADES IMPLEMENTADAS:
-----------------------------
- Ray casting básico
- Intersecção com esferas
- Intersecção com torus
- Modelo de iluminação Phong
- Sombras (shadow feelers)
- Reflexões
- Transparência/refração

Baseado no material da Aula 14 - Ray Tracing
Disciplina: CPS751 - Computação Gráfica
"""

import math
import time

import numpy as np
from PIL import Image

# Configurações da imagem
WIDTH = 800
HEIGHT = 600
CAMERA_POS = np.array([0, 0, 4])  # Câmera mais próxima
VIEWPORT_WIDTH = 4
VIEWPORT_HEIGHT = 3

# Constantes
EPSILON = 1e-6
MAX_DEPTH = 5
BACKGROUND_COLOR = np.array([0.2, 0.2, 0.3])


class Material:
    def __init__(self, color, ambient=0.1, diffuse=0.9, specular=0.5, shininess=50, reflection=0.0, transparency=0.0,
                 refraction_index=1.0):
        self.color = np.array(color)
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection
        self.transparency = transparency
        self.refraction_index = refraction_index


class Light:
    def __init__(self, position, color, intensity=1.0):
        self.position = np.array(position)
        self.color = np.array(color)
        self.intensity = intensity


class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray_origin, ray_dir):
        """Calcula intersecção do raio com a esfera"""
        oc = ray_origin - self.center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - self.radius * self.radius

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # Retorna a menor distância positiva
        if t1 > EPSILON:
            return t1
        elif t2 > EPSILON:
            return t2
        return None

    def normal_at(self, point):
        """Calcula a normal na superfície da esfera"""
        return (point - self.center) / self.radius


class Torus:
    def __init__(self, center, major_radius, minor_radius, material):
        self.center = np.array(center)
        self.major_radius = major_radius  # R - raio maior
        self.minor_radius = minor_radius  # r - raio menor
        self.material = material

    def intersect(self, ray_origin, ray_dir):
        """Calcula intersecção do raio com o torus"""
        # Translada o raio para que o torus esteja na origem
        ox, oy, oz = ray_origin - self.center
        dx, dy, dz = ray_dir

        # Coeficientes da equação quártica
        R = self.major_radius
        r = self.minor_radius

        # Precalcular alguns valores
        sum_d_sqr = dx * dx + dy * dy + dz * dz
        e = ox * ox + oy * oy + oz * oz - R * R - r * r
        f = ox * dx + oy * dy + oz * dz
        four_a_sqr = 4.0 * R * R

        # Coeficientes da equação quártica: At^4 + Bt^3 + Ct^2 + Dt + E = 0
        A = sum_d_sqr * sum_d_sqr
        B = 4.0 * sum_d_sqr * f
        C = 2.0 * sum_d_sqr * e + 4.0 * f * f + four_a_sqr * dz * dz
        D = 4.0 * f * e + 2.0 * four_a_sqr * oz * dz
        E = e * e - four_a_sqr * (r * r - oz * oz)

        # Resolve a equação quártica
        roots = self.solve_quartic(A, B, C, D, E)

        # Encontra a menor raiz positiva
        min_t = float('inf')
        for t in roots:
            if t > EPSILON and t < min_t:
                min_t = t

        return min_t if min_t != float('inf') else None

    def solve_quartic(self, a, b, c, d, e):
        """Resolve equação quártica usando método de Ferrari"""
        if abs(a) < EPSILON:
            return self.solve_cubic(b, c, d, e)

        # Normaliza os coeficientes
        b /= a
        c /= a
        d /= a
        e /= a

        # Substitui x = t - b/4 para eliminar o termo cúbico
        p = c - 3 * b * b / 8
        q = b * b * b / 8 - b * c / 2 + d
        r = -3 * b * b * b * b / 256 + b * b * c / 16 - b * d / 4 + e

        # Caso especial: equação biquadrática
        if abs(q) < EPSILON:
            return self.solve_biquadratic(p, r, b / 4)

        # Resolve a cúbica resolvente
        cubic_roots = self.solve_cubic(1, p / 2, (p * p - 4 * r) / 16, -q * q / 64)

        roots = []
        for y in cubic_roots:
            if y > EPSILON:
                # Verificar se p + 2 * y é positivo antes de calcular sqrt
                term1 = p + 2 * y
                if term1 < 0:
                    continue

                sqrt_y = math.sqrt(y)
                sqrt_term = math.sqrt(term1)

                if abs(sqrt_term) < EPSILON:
                    continue

                sign = 1 if q > 0 else -1

                # Verificar se o termo dentro da segunda raiz é positivo
                term2 = p - 2 * y + sign * 2 * q / (sqrt_y * sqrt_term)
                if term2 < 0:
                    continue

                sqrt_other = math.sqrt(term2)

                # Quatro possíveis raízes
                roots.extend([
                    sqrt_y + sqrt_other - b / 4,
                    sqrt_y - sqrt_other - b / 4,
                    -sqrt_y + sqrt_other - b / 4,
                    -sqrt_y - sqrt_other - b / 4
                ])
                break

        return [r for r in roots if isinstance(r, (int, float)) and not math.isnan(r)]

    def solve_cubic(self, a, b, c, d):
        """Resolve equação cúbica"""
        if abs(a) < EPSILON:
            return self.solve_quadratic(b, c, d)

        # Normaliza
        b /= a
        c /= a
        d /= a

        # Substitui x = t - b/3
        p = c - b * b / 3
        q = 2 * b * b * b / 27 - b * c / 3 + d

        discriminant = q * q / 4 + p * p * p / 27

        roots = []
        if discriminant > 0:
            sqrt_disc = math.sqrt(discriminant)
            u = (-q / 2 + sqrt_disc) ** (1 / 3) if (-q / 2 + sqrt_disc) >= 0 else -(abs(-q / 2 + sqrt_disc) ** (1 / 3))
            v = (-q / 2 - sqrt_disc) ** (1 / 3) if (-q / 2 - sqrt_disc) >= 0 else -(abs(-q / 2 - sqrt_disc) ** (1 / 3))
            roots.append(u + v - b / 3)
        else:
            if abs(p) < EPSILON:
                roots.append(-b / 3)
            else:
                m = 2 * math.sqrt(-p / 3)
                theta = math.acos(3 * q / (p * m))
                roots.extend([
                    m * math.cos(theta / 3) - b / 3,
                    m * math.cos((theta + 2 * math.pi) / 3) - b / 3,
                    m * math.cos((theta + 4 * math.pi) / 3) - b / 3
                ])

        return roots

    def solve_quadratic(self, a, b, c):
        """Resolve equação quadrática"""
        if abs(a) < EPSILON:
            return [-c / b] if abs(b) > EPSILON else []

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return []

        sqrt_disc = math.sqrt(discriminant)
        return [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]

    def solve_biquadratic(self, p, r, offset):
        """Resolve equação biquadrática t^4 + pt^2 + r = 0"""
        discriminant = p * p - 4 * r
        if discriminant < 0:
            return []

        sqrt_disc = math.sqrt(discriminant)
        y1 = (-p + sqrt_disc) / 2
        y2 = (-p - sqrt_disc) / 2

        roots = []
        if y1 >= 0:
            sqrt_y1 = math.sqrt(y1)
            roots.extend([sqrt_y1 - offset, -sqrt_y1 - offset])
        if y2 >= 0:
            sqrt_y2 = math.sqrt(y2)
            roots.extend([sqrt_y2 - offset, -sqrt_y2 - offset])

        return roots

    def normal_at(self, point):
        """Calcula a normal na superfície do torus"""
        x, y, z = point - self.center
        R = self.major_radius
        r = self.minor_radius

        # Distância do ponto ao eixo Z
        rho = math.sqrt(x * x + y * y)

        if rho < EPSILON:
            # Ponto no eixo Z - normal aponta radialmente
            return np.array([1, 0, 0])

        # Ponto no círculo maior mais próximo
        circle_x = R * x / rho
        circle_y = R * y / rho

        # Normal aponta do círculo maior para o ponto
        normal = np.array([
            x - circle_x,
            y - circle_y,
            z
        ])

        length = np.linalg.norm(normal)
        return normal / length if length > EPSILON else np.array([0, 0, 1])


class RayTracer:
    def __init__(self):
        self.objects = []
        self.lights = []

    def add_object(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        self.lights.append(light)

    def intersect_scene(self, ray_origin, ray_dir):
        """Encontra a intersecção mais próxima com a cena"""
        closest_t = float('inf')
        closest_object = None

        for obj in self.objects:
            t = obj.intersect(ray_origin, ray_dir)
            if t is not None and t < closest_t:
                closest_t = t
                closest_object = obj

        return (closest_t, closest_object) if closest_object else (None, None)

    def is_in_shadow(self, point, light_pos):
        """Verifica se um ponto está na sombra (shadow feeler)"""
        light_dir = light_pos - point
        light_distance = np.linalg.norm(light_dir)
        light_dir /= light_distance

        # Move um pouco para evitar auto-intersecção
        shadow_ray_origin = point + light_dir * EPSILON

        t, obj = self.intersect_scene(shadow_ray_origin, light_dir)
        return t is not None and t < light_distance

    def reflect(self, incident, normal):
        """Calcula a direção de reflexão"""
        return incident - 2 * np.dot(incident, normal) * normal

    def refract(self, incident, normal, eta):
        """Calcula a direção de refração"""
        cos_i = -np.dot(incident, normal)
        sin_t2 = eta * eta * (1.0 - cos_i * cos_i)

        if sin_t2 > 1.0:  # Reflexão total interna
            return None

        cos_t = math.sqrt(1.0 - sin_t2)
        return eta * incident + (eta * cos_i - cos_t) * normal

    def phong_lighting(self, point, normal, view_dir, material):
        """Calcula iluminação usando modelo de Phong"""
        color = material.color * material.ambient

        for light in self.lights:
            # Verifica sombras
            if self.is_in_shadow(point, light.position):
                continue

            # Direção da luz
            light_dir = light.position - point
            light_distance = np.linalg.norm(light_dir)
            light_dir /= light_distance

            # Componente difusa
            lambertian = max(0, np.dot(normal, light_dir))
            color += material.color * material.diffuse * lambertian * light.color * light.intensity

            # Componente especular
            if lambertian > 0:
                reflect_dir = self.reflect(-light_dir, normal)
                specular = max(0, np.dot(view_dir, reflect_dir)) ** material.shininess
                color += material.specular * specular * light.color * light.intensity

        return np.clip(color, 0, 1)

    def trace_ray(self, ray_origin, ray_dir, depth=0):
        """Traça um raio através da cena"""
        if depth > MAX_DEPTH:
            return BACKGROUND_COLOR

        t, obj = self.intersect_scene(ray_origin, ray_dir)
        if obj is None:
            return BACKGROUND_COLOR

        # Ponto de intersecção
        hit_point = ray_origin + t * ray_dir
        normal = obj.normal_at(hit_point)
        view_dir = -ray_dir

        # Cor local usando Phong
        local_color = self.phong_lighting(hit_point, normal, view_dir, obj.material)

        # Reflexão
        reflected_color = np.array([0, 0, 0])
        if obj.material.reflection > 0:
            reflect_dir = self.reflect(ray_dir, normal)
            reflect_origin = hit_point + normal * EPSILON
            reflected_color = self.trace_ray(reflect_origin, reflect_dir, depth + 1)

        # Refração/transparência
        transmitted_color = np.array([0, 0, 0])
        if obj.material.transparency > 0:
            eta = 1.0 / obj.material.refraction_index
            if np.dot(ray_dir, normal) > 0:  # Saindo do objeto
                normal = -normal
                eta = obj.material.refraction_index

            refract_dir = self.refract(ray_dir, normal, eta)
            if refract_dir is not None:
                refract_origin = hit_point - normal * EPSILON
                transmitted_color = self.trace_ray(refract_origin, refract_dir, depth + 1)

        # Combina as cores
        final_color = (1 - obj.material.reflection - obj.material.transparency) * local_color
        final_color += obj.material.reflection * reflected_color
        final_color += obj.material.transparency * transmitted_color

        return np.clip(final_color, 0, 1)

    def render(self, width, height):
        """Renderiza a cena"""
        image = np.zeros((height, width, 3))

        print(f"Renderizando imagem {width}x{height}...")
        start_time = time.time()

        for y in range(height):
            if y % 50 == 0:
                print(f"Linha {y}/{height}")

            for x in range(width):
                # Converte coordenadas da tela para o espaço do mundo
                u = (x - width / 2) / width * VIEWPORT_WIDTH
                v = (height / 2 - y) / height * VIEWPORT_HEIGHT

                # Direção do raio
                ray_dir = np.array([u, v, -1])
                ray_dir = ray_dir / np.linalg.norm(ray_dir)

                # Traça o raio
                color = self.trace_ray(CAMERA_POS, ray_dir)
                image[y, x] = color

        end_time = time.time()
        print(f"Renderização concluída em {end_time - start_time:.2f} segundos")

        return image


def create_sphere_scene():
    """Cria uma cena com esferas"""
    raytracer = RayTracer()

    # Materiais
    red_material = Material([0.8, 0.2, 0.2], reflection=0.3)
    blue_material = Material([0.2, 0.2, 0.8], reflection=0.5)
    green_material = Material([0.2, 0.8, 0.2], transparency=0.7, refraction_index=1.5)
    mirror_material = Material([0.9, 0.9, 0.9], reflection=0.8)

    # Esferas
    raytracer.add_object(Sphere([0, 0, 0], 1, red_material))
    raytracer.add_object(Sphere([-2, 0, -1], 0.8, blue_material))
    raytracer.add_object(Sphere([2, 0, -1], 0.8, green_material))
    raytracer.add_object(Sphere([0, -2, -2], 0.6, mirror_material))

    # Luzes
    raytracer.add_light(Light([3, 3, 3], [1, 1, 1], 0.8))
    raytracer.add_light(Light([-3, 2, 2], [0.8, 0.8, 1], 0.6))

    return raytracer


def create_torus_scene():
    """Cria uma cena com torus e esferas"""
    raytracer = RayTracer()

    # Materiais
    torus_material = Material([0.8, 0.4, 0.2], reflection=0.3, specular=0.7, shininess=30)
    sphere_material = Material([0.3, 0.7, 0.9], transparency=0.8, refraction_index=1.5)
    metal_material = Material([0.9, 0.9, 0.9], reflection=0.8, specular=0.9)
    red_material = Material([0.8, 0.2, 0.2], reflection=0.2, specular=0.6)

    # Torus principal - formato de rosquinha
    raytracer.add_object(Torus([0, 0, -1], 1.2, 0.4, torus_material))

    # Esferas para complementar a cena
    raytracer.add_object(Sphere([0, 0, -1], 0.3, sphere_material))  # Centro do torus
    raytracer.add_object(Sphere([2.5, 0, -1], 0.5, metal_material))  # Direita
    raytracer.add_object(Sphere([-2.5, 0, -1], 0.5, red_material))  # Esquerda
    raytracer.add_object(Sphere([0, 2.5, -1], 0.4, Material([0.2, 0.8, 0.2], transparency=0.6)))  # Acima

    # Luzes posicionadas para destacar a forma do torus
    raytracer.add_light(Light([4, 4, 2], [1, 1, 1], 0.8))
    raytracer.add_light(Light([-3, 3, 2], [0.9, 0.9, 1], 0.6))
    raytracer.add_light(Light([0, -3, 3], [1, 0.9, 0.8], 0.5))

    return raytracer


def create_multi_torus_scene():
    """Cria uma cena com múltiplos torus de diferentes tamanhos"""
    raytracer = RayTracer()

    # Materiais diversos
    gold_material = Material([0.9, 0.7, 0.2], reflection=0.6, specular=0.8, shininess=40)
    silver_material = Material([0.8, 0.8, 0.9], reflection=0.7, specular=0.9, shininess=50)
    copper_material = Material([0.7, 0.4, 0.2], reflection=0.4, specular=0.7, shininess=30)
    glass_material = Material([0.9, 0.9, 1.0], transparency=0.8, refraction_index=1.5, reflection=0.1)

    # Torus principal grande
    raytracer.add_object(Torus([0, 0, -2], 1.5, 0.3, gold_material))

    # Torus menor à esquerda
    raytracer.add_object(Torus([-3, 0, -1], 0.8, 0.25, silver_material))

    # Torus menor à direita
    raytracer.add_object(Torus([3, 0, -1], 0.8, 0.25, copper_material))

    # Torus pequeno acima
    raytracer.add_object(Torus([0, 2, 0], 0.6, 0.2, glass_material))

    # Esferas decorativas
    raytracer.add_object(Sphere([0, 0, -2], 0.2, Material([1, 1, 1], transparency=0.9, refraction_index=1.5)))
    raytracer.add_object(Sphere([0, -3, -2], 0.4, Material([0.8, 0.2, 0.8], reflection=0.5)))

    # Iluminação dramática
    raytracer.add_light(Light([5, 5, 3], [1, 1, 1], 0.9))
    raytracer.add_light(Light([-4, 3, 2], [0.8, 0.9, 1], 0.7))
    raytracer.add_light(Light([0, -4, 4], [1, 0.8, 0.6], 0.6))

    return raytracer


def create_sphere_cluster_scene():
    """Cria uma cena com cluster de esferas de diferentes materiais"""
    raytracer = RayTracer()

    # Materiais variados
    materials = [
        Material([0.8, 0.2, 0.2], reflection=0.3, specular=0.6),  # Vermelho
        Material([0.2, 0.8, 0.2], reflection=0.4, specular=0.7),  # Verde
        Material([0.2, 0.2, 0.8], reflection=0.5, specular=0.8),  # Azul
        Material([0.8, 0.8, 0.2], reflection=0.2, specular=0.5),  # Amarelo
        Material([0.8, 0.2, 0.8], reflection=0.6, specular=0.9),  # Magenta
        Material([0.2, 0.8, 0.8], transparency=0.7, refraction_index=1.5),  # Ciano transparente
        Material([0.9, 0.9, 0.9], reflection=0.8, specular=0.9),  # Espelho
        Material([0.5, 0.5, 0.5], transparency=0.6, refraction_index=1.3, reflection=0.3),  # Vidro
    ]

    # Esferas em formação circular
    import math
    num_spheres = 8
    radius = 2.5
    for i in range(num_spheres):
        angle = 2 * math.pi * i / num_spheres
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = -1 + 0.3 * math.sin(3 * angle)  # Variação na profundidade
        sphere_radius = 0.4 + 0.2 * math.sin(2 * angle)  # Variação no tamanho

        raytracer.add_object(Sphere([x, y, z], sphere_radius, materials[i]))

    # Esfera central
    raytracer.add_object(Sphere([0, 0, -1], 0.8, Material([1, 1, 1], transparency=0.9, refraction_index=1.5)))

    # Iluminação múltipla
    raytracer.add_light(Light([4, 4, 4], [1, 1, 1], 0.8))
    raytracer.add_light(Light([-4, 4, 4], [1, 0.8, 0.6], 0.6))
    raytracer.add_light(Light([0, -4, 4], [0.6, 0.8, 1], 0.7))

    return raytracer


def create_torus_sphere_mix_scene():
    """Cria uma cena mista com torus e esferas em composição artística"""
    raytracer = RayTracer()

    # Materiais artísticos
    marble_material = Material([0.9, 0.9, 0.8], reflection=0.3, specular=0.8, shininess=60)
    bronze_material = Material([0.7, 0.5, 0.3], reflection=0.4, specular=0.7, shininess=40)
    emerald_material = Material([0.2, 0.8, 0.3], transparency=0.6, refraction_index=1.6, reflection=0.3)
    ruby_material = Material([0.8, 0.2, 0.2], transparency=0.4, refraction_index=1.7, reflection=0.4)

    # Torus principal vertical (rotacionado)
    raytracer.add_object(Torus([0, 0, -1.5], 1.3, 0.4, bronze_material))

    # Torus horizontal menor
    raytracer.add_object(Torus([0, 0, -1.5], 0.8, 0.2, marble_material))

    # Esferas grandes
    raytracer.add_object(Sphere([-2, 1, -1], 0.7, emerald_material))
    raytracer.add_object(Sphere([2, -1, -1], 0.6, ruby_material))

    # Esferas pequenas decorativas
    raytracer.add_object(Sphere([0, 0, -1.5], 0.3, Material([1, 1, 1], transparency=0.95, refraction_index=1.5)))
    raytracer.add_object(Sphere([1.5, 1.5, 0], 0.25, Material([0.8, 0.8, 0.2], reflection=0.6)))
    raytracer.add_object(Sphere([-1.5, -1.5, 0], 0.25, Material([0.2, 0.8, 0.8], reflection=0.5)))

    # Iluminação cinematográfica
    raytracer.add_light(Light([3, 4, 3], [1, 1, 1], 0.9))
    raytracer.add_light(Light([-3, 2, 2], [0.8, 0.9, 1], 0.6))
    raytracer.add_light(Light([0, -3, 4], [1, 0.7, 0.5], 0.5))

    return raytracer


def create_abstract_scene():
    """Cria uma cena abstrata com formas geométricas complexas"""
    raytracer = RayTracer()

    # Materiais futuristas
    neon_blue = Material([0.2, 0.6, 1.0], reflection=0.5, specular=0.9, shininess=80)
    neon_pink = Material([1.0, 0.2, 0.6], reflection=0.4, specular=0.8, shininess=70)
    chrome = Material([0.8, 0.8, 0.9], reflection=0.9, specular=1.0, shininess=100)
    plasma = Material([0.9, 0.4, 0.9], transparency=0.7, refraction_index=1.4, reflection=0.2)

    # Arranjo de torus em diferentes orientações
    raytracer.add_object(Torus([0, 0, -2], 1.0, 0.3, neon_blue))
    raytracer.add_object(Torus([0, 0, -2], 0.7, 0.15, neon_pink))

    # Esferas em posições estratégicas
    raytracer.add_object(Sphere([0, 0, -2], 0.4, plasma))
    raytracer.add_object(Sphere([2, 0, -1], 0.5, chrome))
    raytracer.add_object(Sphere([-2, 0, -1], 0.5, chrome))
    raytracer.add_object(Sphere([0, 2, -1], 0.4, neon_blue))
    raytracer.add_object(Sphere([0, -2, -1], 0.4, neon_pink))

    # Pequenas esferas orbitais
    import math
    for i in range(6):
        angle = 2 * math.pi * i / 6
        x = 1.5 * math.cos(angle)
        y = 1.5 * math.sin(angle)
        z = -1 + 0.5 * math.sin(angle * 3)
        raytracer.add_object(Sphere([x, y, z], 0.15, Material([1, 1, 1], transparency=0.8, refraction_index=1.5)))

    # Iluminação colorida
    raytracer.add_light(Light([4, 4, 4], [1, 1, 1], 0.8))
    raytracer.add_light(Light([-4, 4, 4], [0.2, 0.6, 1], 0.6))
    raytracer.add_light(Light([0, -4, 4], [1, 0.2, 0.6], 0.7))

    return raytracer


def main():
    """Função principal"""
    print("=== RAY TRACER CPS751 ===")
    print("Gerando múltiplas cenas...")

    # Lista de cenas para renderizar
    scenes = [
        ("esfera_scene.png", "Esferas clássicas", create_sphere_scene),
        ("torus_scene.png", "Torus com esferas", create_torus_scene),
        ("multi_torus_scene.png", "Múltiplos torus metálicos", create_multi_torus_scene),
        ("sphere_cluster_scene.png", "Cluster de esferas coloridas", create_sphere_cluster_scene),
        ("torus_sphere_mix_scene.png", "Composição artística", create_torus_sphere_mix_scene),
        ("abstract_scene.png", "Cena abstrata futurista", create_abstract_scene),
    ]

    print(f"\nSerão renderizadas {len(scenes)} cenas diferentes:")
    for i, (filename, description, _) in enumerate(scenes, 1):
        print(f"  {i}. {description} -> {filename}")

    # Renderiza todas as cenas
    for i, (filename, description, scene_func) in enumerate(scenes, 1):
        print(f"\n{i}. Renderizando {description}...")
        scene = scene_func()
        image = scene.render(WIDTH, HEIGHT)

        # Salva a imagem
        img = Image.fromarray((image * 255).astype(np.uint8))
        img.save(filename)
        print(f"Imagem '{filename}' salva com sucesso!")

    print("\n=== RENDERIZAÇÃO CONCLUÍDA ===")
    print("Funcionalidades implementadas:")
    print("- Intersecção com esferas")
    print("- Intersecção com torus")
    print("- Modelo de iluminação Phong")
    print("- Sombras (shadow feelers)")
    print("- Reflexões")
    print("- Transparência/refração")
    print("\nCenas geradas:")
    for filename, description, _ in scenes:
        print(f"  ✓ {filename} - {description}")
    print("\nVerifique as imagens geradas!")


if __name__ == "__main__":
    main()
