//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kanizsai Szilárd Erik
// 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";


enum MetarialType { ROUGH, REFLECTIVE, REFLECTIVEWATER };

struct Material {
	vec3 ka, kd, ks, F0;
	float  shininess;
	MetarialType type;
	vec3 N, K;
	Material(MetarialType t) {
		type = t;
	}
};

struct RoughMetarial : Material {

	RoughMetarial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

const vec3 one(1, 1, 1);
vec3 operator/(vec3 a, vec3 b) { return vec3(a.x / b.x, a.y / b.y, a.z / b.z); }

struct ReflectiveMetarial : Material {

	ReflectiveMetarial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
		N = n;
		K = kappa;
	}

};

struct ReflectiveWaterMetarial : Material {

	ReflectiveWaterMetarial(vec3 n, vec3 kappa) : Material(REFLECTIVEWATER) {
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
		N = n;
		K = kappa;
	}
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	bool out = true;
	Ray(vec3 _start, vec3 _dir, bool myOut) { start = _start; dir = normalize(_dir), out = myOut; }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Plane : public Intersectable {
	vec3 normal;
	float d;

	Plane(const vec3& _normal, float _d, Material* _material) {
		normal = normalize(_normal);
		d = _d;
		material = _material;
	}

	Hit intersect(const Ray& ray) override {
		Hit hit;
		float denom = dot(normal, ray.dir);
		if (fabs(denom) > 0.0001f) {
			float t = (d - dot(normal, ray.start)) / denom;
			if (t >= 0) {
				vec3 hitPosition = ray.start + ray.dir * t;

				if (fabs(hitPosition.x) <= 10.0 && fabs(hitPosition.z) <= 10.0) {
					hit.t = t;
					hit.position = hitPosition;
					hit.normal = normal;
					hit.material = material;
					int checkerWidth = 1;
					int ix = int(floor(hit.position.x / checkerWidth));
					int iz = int(floor(hit.position.z / checkerWidth));
					if ((ix + iz) % 2 == 0)
						hit.material = new RoughMetarial(vec3(0.3, 0.3, 0.3), vec3(0.0, 0.0, 0.0), 0);
					else
						hit.material = new RoughMetarial(vec3(0.0, 0.1, 0.3), vec3(0.0, 0.0, 0.0), 0);
				}
			}
		}
		return hit;
	}
};

struct Cylinder : public Intersectable {
	vec3 basePoint;
	vec3 axis;
	float radius;
	float height;

	Cylinder(const vec3& _basePoint, const vec3& _axis, float _radius, float _height, Material* _material) {
		basePoint = _basePoint;
		axis = normalize(_axis);
		radius = _radius;
		height = _height;
		material = _material;
	}

	Hit intersect(const Ray& ray) override {
		Hit hit;

		vec3 deltaP = ray.start - basePoint;
		vec3 m = cross(ray.dir, axis);
		vec3 n = cross(deltaP, axis);
		float a = dot(m, m);
		float b = 2 * dot(m, n);
		float c = dot(n, n) - radius * radius * dot(axis, axis);

		float discr = b * b - 4 * a * c;
		if (discr < 0) return hit;

		float sqrt_discr = sqrt(discr);
		float t1 = (-b - sqrt_discr) / (2 * a);
		float t2 = (-b + sqrt_discr) / (2 * a);

		float t = (t1 < t2) ? t1 : t2;
		if (t < 0) t = (t1 < t2) ? t2 : t1;
		if (t < 0) return hit;

		vec3 hitPoint = ray.start + ray.dir * t;
		float projection = dot((hitPoint - basePoint), axis);
		if (projection < 0 || projection > height) return hit;

		hit.t = t;
		hit.position = hitPoint;
		hit.normal = normalize(cross(cross(axis, hitPoint - basePoint), axis));
		hit.material = material;

		return hit;
	}
};

struct Cone : public Intersectable {
	vec3 apex;
	vec3 axis;
	float angle;
	float height;

	Cone(const vec3& _apex, const vec3& _axis, float _angle, float _height, Material* _material) {
		apex = _apex;
		axis = normalize(_axis);
		angle = _angle;
		height = _height;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		vec3 CO = ray.start - apex;

		float A = dot(ray.dir, axis) * dot(ray.dir, axis) - cosf(angle) * cosf(angle);
		float B = 2 * (dot(ray.dir, axis) * dot(CO, axis) - dot(ray.dir, CO) * cosf(angle) * cosf(angle));
		float C = dot(CO, axis) * dot(CO, axis) - dot(CO, CO) * cosf(angle) * cosf(angle);

		float discriminant = B * B - 4 * A * C;

		if (discriminant < 0) return hit;


		float sqrtDiscriminant = sqrt(discriminant);
		float t1 = (-B - sqrtDiscriminant) / (2 * A);
		float t2 = (-B + sqrtDiscriminant) / (2 * A);


		float t = t1 < t2 ? t1 : t2;
		if (t < 0) t = t1 > t2 ? t1 : t2;
		if (t < 0) return hit;


		vec3 intersectPoint = ray.start + ray.dir * t;
		float heightAtPoint = dot(intersectPoint - apex, axis);
		if (heightAtPoint < 0 || heightAtPoint > height) return hit;


		hit.t = t;
		hit.position = intersectPoint;
		hit.normal = normalize(2 * dot((intersectPoint - apex), axis) * axis - 2 * (intersectPoint - apex) * cosf(angle) * cosf(angle));
		hit.material = material;

		return hit;
	}

};

class Camera {
	vec3 eye, lookat, right, up;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
		eye = _eye;
		lookat = _lookat;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir, true);
	}
};

struct Light {
	vec3 direction;
	vec3 Le;
	Light(vec3 _direction, vec3 _Le) {
		direction = normalize(_direction);
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;

	vec3 La;

public:
	Camera camera;

	void build(float rotateX, float rotateY) {
		vec3 eye = vec3(rotateX, 1, rotateY), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		Material* checkerMaterial = new RoughMetarial(vec3(0.3, 0.2, 0.1), vec3(2, 2, 2), 50);
		objects.push_back(new Plane(vec3(0, 1, 0), -1, checkerMaterial));

		Material* checkerMaterial3 = new ReflectiveMetarial(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9));
		Material* checkerMaterial4 = new ReflectiveWaterMetarial(vec3(1.3, 1.3, 1.3), vec3(0, 0, 0));

		objects.push_back(new Cylinder(vec3(1, -1, 0), vec3(0.1, 1, 0), 0.3, 2, checkerMaterial3));
		objects.push_back(new Cylinder(vec3(0, -1, -0.8), vec3(-0.2, 1, -0.1), 0.3, 2, checkerMaterial4));
		objects.push_back(new Cylinder(vec3(-1, -1, 0), vec3(0, 1, 0.1), 0.3, 2, checkerMaterial));

		Material* checkerMaterial1 = new RoughMetarial(vec3(0.1, 0.2, 0.3), vec3(2, 2, 2), 100);
		Material* checkerMaterial2 = new RoughMetarial(vec3(0.3, 0, 0.2), vec3(2, 2, 2), 20);
		objects.push_back(new Cone(vec3(0, 1, 0), vec3(-0.1, -1, -0.05), 0.2, 2, checkerMaterial1));
		objects.push_back(new Cone(vec3(0, 1, 0.8), vec3(0.2, -1, 0), 0.2, 2, checkerMaterial2));

	}

	void render(std::vector<vec4>& image) {
		for (unsigned int Y = 0; Y < windowHeight; Y++) {

			for (unsigned int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y), 2);
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray);
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 refract(vec3 V, vec3 N, float ns) {
		float cosa = -dot(V, N);
		float disc = 1.0f - (1.0f - cosa * cosa) / ns / ns;
		if (disc < 0) return vec3(0, 0, 0);
		return V / ns + N * (cosa / ns - sqrtf(disc));
	}

	vec3 reflect(vec3 V, vec3 N) {
		return V - N * dot(N, V) * 2.0f;
	}

	vec3 Fresnel(vec3 V, vec3 N, vec3 n, vec3 kappa) {
		float cosa = -dot(V, N);
		vec3 one(1.0f, 1.0f, 1.0f);
		vec3 F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
		return F0 + (one - F0) * pow(1.0f - cosa, 5);
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) {
			return La;
		}
		Hit hit = firstIntersect(ray);
		vec3 outRadiance(0, 0, 0);
		if (hit.t < 0) return La;

		if (hit.material->type == ROUGH) {

			outRadiance = hit.material->ka * La;
			for (Light* light : lights) {
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction, true);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
				}
			}
		}

		if (hit.material->type == REFLECTIVE || hit.material->type == REFLECTIVEWATER) {
			vec3 reflectedDir = reflect(ray.dir, hit.normal);
			vec3 fresnelEffect = Fresnel(ray.dir, hit.normal, hit.material->N, hit.material->K);
			outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir, true), depth + 1) * fresnelEffect;
		}

		if (hit.material->type == REFLECTIVEWATER) {
			float ior = (ray.out) ? hit.material->N.x : 1 / hit.material->N.x;

			vec3 refractedDir = refract(ray.dir, hit.normal, ior);
			if (length(refractedDir) > 0) {
				vec3 fresnelEffect = Fresnel(ray.dir, hit.normal, hit.material->N, hit.material->K);
				outRadiance = outRadiance + trace(Ray(hit.position - hit.normal * epsilon, refractedDir, false), depth + 1) * (vec3(1, 1, 1) - fresnelEffect);
			}
		}
		return outRadiance;
	}
};

GPUProgram gpuProgram;
Scene scene;


class FullScreenTexturedQuad {
	unsigned int vao, vbo[2];
	unsigned int textureId;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
	{

		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		float uvs[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(2, vbo);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), &vertexCoords[0], GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

		glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), &uvs[0], GL_DYNAMIC_DRAW);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0,
			GL_RGBA, GL_FLOAT, &image[0]);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build(0, 4);

	std::vector<vec4> image(windowHeight * windowWidth);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();

}
void elforgat(float elso, float masodik) {

	std::vector<vec4> image(windowHeight * windowWidth);
	scene.camera.set(vec3(elso, 1, masodik), vec3(0, 0, 0), vec3(0, 1, 0), 45 * M_PI / 180);
	scene.render(image);
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
}

int valto = 0;

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'a') {

		switch (valto)
		{
		case 0:
			elforgat(-sqrtf(8), sqrtf(8));
			valto++;
			break;

		case 1:
			elforgat(-4, 0);
			valto++;
			break;

		case 2:
			elforgat(-sqrtf(8), -sqrtf(8));
			valto++;
			break;

		case 3:
			elforgat(0, -4);
			valto++;
			break;

		case 4:
			elforgat(sqrtf(8), -sqrtf(8));
			valto++;
			break;

		case 5:
			elforgat(4, 0);
			valto++;
			break;

		case 6:
			elforgat(sqrtf(8), sqrtf(8));
			valto++;
			break;

		case 7:
			elforgat(0, 4);
			valto = 0;
			break;

		default:
			break;
		}
		glutPostRedisplay();
	}


}

void onKeyboardUp(unsigned char key, int pX, int pY) {

}

void onMouse(int button, int state, int pX, int pY) {
}

void onMouseMotion(int pX, int pY) {
}

void onIdle() {
}
