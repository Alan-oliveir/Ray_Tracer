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

import numpy as np
import math
from PIL import Image
import time

# Configurações da imagem
WIDTH = 800
HEIGHT = 600
CAMERA_POS = np.array([0, 0, 5])
VIEWPORT_WIDTH = 4
VIEWPORT_HEIGHT = 3

# Constantes
EPSILON = 1e-6
MAX_DEPTH = 5
BACKGROUND_COLOR = np.array([0.2, 0.2, 0.3])

class Material:
    def __init__(self, color, ambient=0.1, diffuse=0.9, specular=0.5, shininess=50, reflection=0.0, transparency=0.0, refraction_index=1.0):
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
        sum_d_sqr = dx*dx + dy*dy + dz*dz
        e = ox*ox + oy*oy + oz*oz - R*R - r*r
        f = ox*dx + oy*dy + oz*dz
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
        p = c - 3*b*b/8
        q = b*b*b/8 - b*c/2 + d
        r = -3*b*b*b*b/256 + b*b*c/16 - b*d/4 + e
        
        # Caso especial: equação biquadrática
        if abs(q) < EPSILON:
            return self.solve_biquadratic(p, r, b/4)
        
        # Resolve a cúbica resolvente
        cubic_roots = self.solve_cubic(1, p/2, (p*p - 4*r)/16, -q*q/64)
        
        roots = []
        for y in cubic_roots:
            if y > EPSILON:
                sqrt_y = math.sqrt(y)
                sqrt_term = math.sqrt(p + 2*y)
                
                if abs(sqrt_term) < EPSILON:
                    continue
                
                sign = 1 if q > 0 else -1
                sqrt_other = math.sqrt(abs(p - 2*y + sign*2*q/(sqrt_y*sqrt_term)))
                
                # Quatro possíveis raízes
                roots.extend([
                    sqrt_y + sqrt_other - b/4,
                    sqrt_y - sqrt_other - b/4,
                    -sqrt_y + sqrt_other - b/4,
                    -sqrt_y - sqrt_other - b/4
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
        p = c - b*b/3
        q = 2*b*b*b/27 - b*c/3 + d
        
        discriminant = q*q/4 + p*p*p/27
        
        roots = []
        if discriminant > 0:
            sqrt_disc = math.sqrt(discriminant)
            u = (-q/2 + sqrt_disc)**(1/3) if (-q/2 + sqrt_disc) >= 0 else -(abs(-q/2 + sqrt_disc)**(1/3))
            v = (-q/2 - sqrt_disc)**(1/3) if (-q/2 - sqrt_disc) >= 0 else -(abs(-q/2 - sqrt_disc)**(1/3))
            roots.append(u + v - b/3)
        else:
            if abs(p) < EPSILON:
                roots.append(-b/3)
            else:
                m = 2 * math.sqrt(-p/3)
                theta = math.acos(3*q/(p*m))
                roots.extend([
                    m * math.cos(theta/3) - b/3,
                    m * math.cos((theta + 2*math.pi)/3) - b/3,
                    m * math.cos((theta + 4*math.pi)/3) - b/3
                ])
        
        return roots
    
    def solve_quadratic(self, a, b, c):
        """Resolve equação quadrática"""
        if abs(a) < EPSILON:
            return [-c/b] if abs(b) > EPSILON else []
        
        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return []
        
        sqrt_disc = math.sqrt(discriminant)
        return [(-b + sqrt_disc)/(2*a), (-b - sqrt_disc)/(2*a)]
    
    def solve_biquadratic(self, p, r, offset):
        """Resolve equação biquadrática t^4 + pt^2 + r = 0"""
        discriminant = p*p - 4*r
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
        
        # Calcula a normal usando as derivadas parciais
        k = (x*x + y*y + z*z - R*R - self.minor_radius*self.minor_radius) / (2 * R)
        
        normal = np.array([
            x * (1 - k / math.sqrt(x*x + y*y)) if math.sqrt(x*x + y*y) > EPSILON else 0,
            y * (1 - k / math.sqrt(x*x + y*y)) if math.sqrt(x*x + y*y) > EPSILON else 0,
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
                u = (x - width/2) / width * VIEWPORT_WIDTH
                v = (height/2 - y) / height * VIEWPORT_HEIGHT
                
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
    torus_material = Material([0.8, 0.6, 0.2], reflection=0.4, specular=0.8)
    sphere_material = Material([0.6, 0.2, 0.8], transparency=0.6, refraction_index=1.3)
    metal_material = Material([0.7, 0.7, 0.8], reflection=0.9)
    
    # Torus
    raytracer.add_object(Torus([0, 0, 0], 1.5, 0.5, torus_material))
    
    # Esferas
    raytracer.add_object(Sphere([0, 0, 0], 0.4, sphere_material))
    raytracer.add_object(Sphere([2, 1, -1], 0.6, metal_material))
    raytracer.add_object(Sphere([-2, -1, 1], 0.7, Material([0.8, 0.2, 0.2], reflection=0.2)))
    
    # Luzes
    raytracer.add_light(Light([3, 3, 3], [1, 1, 1], 0.9))
    raytracer.add_light(Light([-2, 2, 4], [1, 0.8, 0.8], 0.7))
    
    return raytracer

def main():
    """Função principal"""
    print("=== RAY TRACER CPS751 ===")
    print("Gerando cenas...")
    
    # Cena 1: Esferas
    print("\n1. Renderizando cena com esferas...")
    sphere_scene = create_sphere_scene()
    sphere_image = sphere_scene.render(WIDTH, HEIGHT)
    
    # Salva imagem das esferas
    sphere_img = Image.fromarray((sphere_image * 255).astype(np.uint8))
    sphere_img.save('esfera_scene.png')
    print("✓ Imagem 'esfera_scene.png' salva com sucesso!")
    
    # Cena 2: Torus
    print("\n2. Renderizando cena com torus...")
    torus_scene = create_torus_scene()
    torus_image = torus_scene.render(WIDTH, HEIGHT)
    
    # Salva imagem do torus
    torus_img = Image.fromarray((torus_image * 255).astype(np.uint8))
    torus_img.save('torus_scene.png')
    print("✓ Imagem 'torus_scene.png' salva com sucesso!")
    
    print("\n=== RENDERIZAÇÃO CONCLUÍDA ===")
    print("Funcionalidades implementadas:")
    print("- ✓ Intersecção com esferas")
    print("- ✓ Intersecção com torus")
    print("- ✓ Modelo de iluminação Phong")
    print("- ✓ Sombras (shadow feelers)")
    print("- ✓ Reflexões")
    print("- ✓ Transparência/refração")
    print("\nVerifique as imagens geradas!")

if __name__ == "__main__":
    main()
