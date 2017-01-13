#include "scene.h"
#include "intersect.h"
#include "montecarlo.h"

#include <thread>
using std::thread;

// lookup texture value
vec3f lookup_scaled_texture(vec3f value, image3f* texture, vec2f uv, bool tile = false) {
	// YOUR CODE GOES HERE ----------------------
	if (texture == nullptr) { return value; }
	int i = uv.x * texture->width();
	int j = uv.y * texture->height();
	int i_prime = i + 1;
	int j_prime = j + 1;
	float s = uv[0] * texture->width() - i;
	float t = uv[1] * texture->height() - j;
	if (!tile) {
		i = clamp(i, 0, texture->width() - 1);
		j = clamp(j, 0, texture->height() - 1);
		i_prime = clamp(i_prime, 0, texture->width() - 1);
		j_prime = clamp(j_prime, 0, texture->height() - 1);
	}
	else {
		i = i % texture->width();
		j = j % texture->height();
		i_prime = i_prime % texture->width();
		j_prime = j_prime % texture->height();

		if (i < 0) { i = i + texture->width(); }
		if (j < 0)	{ j = j + texture->height(); }
		if (i_prime < 0) { i_prime = i_prime + texture->width(); }
		if (j_prime < 0) { j_prime = j_prime + texture->height(); }
	}
	//compute bilinear filtering
	value *= texture->at(i, j) * (1 - s) * (1 - t) +
		texture->at(i, j_prime) * (1 - s) * t +
		texture->at(i_prime, j) * s * (1 - t) +
		texture->at(i_prime, j_prime) * s * t;
	return value;
}

// compute the brdf
vec3f eval_brdf(vec3f kd, vec3f ks, float n, vec3f v, vec3f l, vec3f norm, bool microfacet) {
	// YOUR CODE GOES HERE ----------------------
	auto h = normalize(v + l); // placeholder (non-microfacet model)
	if (microfacet) {
		auto d = ((n + 2) / (2 * pif)) * pow(max(0.0f, dot(norm,h)), n);
		auto f = ks + (one3f - ks) * pow((1 - dot(norm, l)), 5);
		auto first_check = min(1.0f, (2 * dot(h, norm) * dot(v, norm)) / dot(v, h));
		auto second_check = min((2 * dot(h, norm) * dot(l, norm)) / dot(l, h),first_check);
		auto g = second_check;
		auto ro = (d * g * f) / (4 * dot(l, norm) * dot(v, norm));
		return ro;
	}
	return kd / pif + ks*(n + 8) / (8 * pif) * pow(max(0.0f, dot(norm, h)), n); // placeholder (non-microfacet model)
}

// evaluate the environment map
vec3f eval_env(vec3f ke, image3f* ke_txt, vec3f dir) {
	// YOUR CODE GOES HERE ----------------------
	if (not ke_txt) return zero3f;
	auto u = atan2(dir.x, dir.z) / (2 * pif);
	auto v = 1 - acos(dir.y) / pif;
	return lookup_scaled_texture(ke, ke_txt, vec2f(u, v),true);
}

// pick a direction according to the cosine (returns direction and its pdf)
pair<vec3f, float> sample_cosine(vec3f norm, vec2f ruv) {
	auto frame = frame_from_z(norm);
	auto l_local = sample_direction_hemispherical_cosine(ruv);
	auto pdf = sample_direction_hemispherical_cosine_pdf(l_local);
	auto l = transform_direction(frame, l_local);
	return{ l, pdf };
}

// pick a direction according to the brdf (returns direction and its pdf)
pair<vec3f, float> sample_brdf(vec3f kd, vec3f ks, float n, vec3f v, vec3f norm, vec2f ruv, float rl) {
	if (ks == zero3f) return sample_cosine(norm, ruv);
	auto frame = frame_from_z(norm);
	auto dw = mean(kd) / (mean(kd) + mean(ks));
	auto v_local = transform_direction_inverse(frame, v);
	auto l_local = zero3f, h_local = zero3f;
	if (rl < dw) {
		l_local = sample_direction_hemispherical_cosine(ruv);
		h_local = normalize(l_local + v_local);
	}
	else {
		h_local = sample_direction_hemispherical_cospower(ruv, n);
		l_local = -v_local + h_local * 2 * dot(v_local, h_local);
	}
	auto l = transform_direction(frame, l_local);
	auto dpdf = sample_direction_hemispherical_cosine_pdf(l_local);
	auto spdf = sample_direction_hemispherical_cospower_pdf(h_local, n) / (4 * dot(v_local, h_local));
	auto pdf = dw * dpdf + (1 - dw) * spdf;
	return{ l, pdf };
}

// compute the color corresponing to a ray by pathtrace
vec3f pathtrace_ray(Scene* scene, ray3f ray, Rng* rng, int depth) {
	// get scene intersection
	auto intersection = intersect(scene, ray);

	// if not hit, return background (looking up the texture by converting the ray direction to latlong around y)
	if (not intersection.hit) {
		auto background = eval_env(scene->background, scene->background_txt, ray.d);
		return background;
	}

	// setup variables for shorter code
	auto pos = intersection.pos;
	auto norm = intersection.norm;
	auto v = -ray.d;

	// compute material values by looking up textures
	// YOUR CODE GOES HERE ----------------------
	auto ke = lookup_scaled_texture(intersection.mat->ke, intersection.mat->ke_txt, intersection.texcoord);
	auto kd = lookup_scaled_texture(intersection.mat->kd, intersection.mat->kd_txt, intersection.texcoord);
	auto ks = lookup_scaled_texture(intersection.mat->ks, intersection.mat->ks_txt, intersection.texcoord);
	auto n = intersection.mat->n;
	auto mf = intersection.mat->microfacet;

	// accumulate color starting with ambient
	auto c = scene->ambient * kd;

	// add emission if on the first bounce
	if (depth == 0) { c += ke; }

	// foreach point light
	for (auto light : scene->lights) {
		// compute light response
		auto cl = light->intensity / (lengthSqr(light->frame.o - pos));
		// compute light direction
		auto l = normalize(light->frame.o - pos);
		// compute the material response (brdf*cos)
		auto brdfcos = max(dot(norm, l), 0.0f) * eval_brdf(kd, ks, n, v, l, norm, mf);
		// multiply brdf and light
		auto shade = cl * brdfcos;
		// check for shadows and accumulate if needed
		if (shade == zero3f) continue;
		// if shadows are enabled
		if (scene->path_shadows) {
			// perform a shadow check and accumulate
			if (not intersect_shadow(scene, ray3f::make_segment(pos, light->frame.o))) c += shade;
		}
		else {
			// else just accumulate
			c += shade;
		}
	}

	// YOUR AREA LIGHT CODE GOES HERE ----------------------
	// foreach surface
	for (auto surface : scene->surfaces) {
		// skip if no emission from surface
		if (surface->mat->ke == zero3f) { continue; }
		// pick a point on the surface, grabbing normal, area and texcoord		
		float surface_area;
		vec3f surface_point;
		vec3f nl;
		vec3f s;
		vec2f surface_texcoord;
		vec2f rand_uv;
		// check if quad
		if (surface->isquad) {
			// generate a 2d random number
			rand_uv = rng->next_vec2f();

			// compute light position, normal, area
			surface_area = 4 * surface->radius * surface->radius;
			surface_point = vec3f(2 * surface->radius * (rand_uv.x - 0.5), 2 * surface->radius * (rand_uv.y - 0.5), 0.0);
			s = transform_point(surface->frame, surface_point);
			nl = transform_normal(surface->frame, vec3f(0, 0, 1));

			// set tex coords as random value got before
			surface_texcoord = rand_uv;
		}
		else {
			// else
			// generate a 2d random number
			rand_uv = rng->next_vec2f();
			// compute light position, normal, area
			surface_area = 4 * surface->radius * surface->radius;
			surface_point = vec3f(2 * surface->radius * (rand_uv.x - 0.5), 2 * surface->radius * (rand_uv.y - 0.5), 0.0);
			s = transform_point(surface->frame, surface_point);
			nl = transform_normal(surface->frame, vec3f(0, 0, 1));
			// set tex coords as random value got before
			surface_texcoord = rand_uv;
		}
		// get light emission from material and texture
		auto surface_ke = lookup_scaled_texture(surface->mat->ke, surface->mat->ke_txt, surface_texcoord);
		// compute light direction
		auto l = normalize(s - pos);
		// compute light response
		auto cl = surface_ke * surface_area *
			max(0.0, -1 * dot(nl, l)) / lengthSqr(s - pos);
		// compute the material response (brdf*cos)
		auto brdfcos = max(dot(norm, l), 0.0f) * eval_brdf(kd, ks, n, v, l, norm, mf);
		// multiply brdf and light
		cl = brdfcos * cl;
		// check for shadows and accumulate if needed
		// if shadows are enabled
		if (scene->path_shadows) {
			// perform a shadow check and accumulate
			if (!intersect_shadow(scene, ray3f::make_segment(pos, s)))
				c += cl;
		}
		else {
			// else just accumulate
			c += cl;
		}
	}
	// YOUR ENVIRONMENT LIGHT CODE GOES HERE ----------------------
	// sample the brdf for environment illumination if the environment is there
	if (scene->background_txt != nullptr) {	
		auto brdf_pair = sample_brdf(kd, ks, n, v, norm, rng->next_vec2f(), rng->next_float());
		// pick direction and pdf
		auto brdf_direction = brdf_pair.first;
		auto brdf_pdf = brdf_pair.second;
		// compute the material response (brdf*cos)
		auto brdfcos = max(dot(norm, brdf_direction), 0.0f) * eval_brdf(kd, ks, n, v, brdf_direction, norm, mf);
		// accumulate recersively scaled by brdf*cos/pdf
		auto cl = (brdfcos / brdf_pdf) * eval_env(scene->background,scene->background_txt,brdf_direction);
		if (scene->path_shadows) {
			// if shadows are enabled
			if (!intersect_shadow(scene, ray3f::ray3f(pos,brdf_direction)))
				c += cl;
				// perform a shadow check and accumulate
		}
		// else
		else {
			// else just accumulate
			c += cl;
		}
	}
	// YOUR INDIRECT ILLUMINATION CODE GOES HERE ----------------------
	// sample the brdf for indirect illumination
	if (depth < scene->path_max_depth) {
		auto indirect_pair = sample_brdf(kd, ks, n, v, norm, rng->next_vec2f(), rng->next_float());
		// pick direction and pdf
		auto indirect_direction = indirect_pair.first;
		auto indirect_pdf = indirect_pair.second;
		// compute the material response (brdf*cos)
		auto brdfcos = max(dot(norm, indirect_direction), 0.0f) * eval_brdf(kd, ks, n, v, indirect_direction, norm, mf);
		// accumulate recersively scaled by brdf*cos/pdf
		c += (brdfcos * pathtrace_ray(scene, ray3f(pos, indirect_direction), rng, depth+1)) / indirect_pdf;
	}
	// return the accumulated color
	return c;
}

// pathtrace an image
void pathtrace(Scene* scene, image3f* image, RngImage* rngs, int offset_row, int skip_row, bool verbose) {
	if (verbose) message("\n  rendering started        ");
	// foreach pixel
	for (auto j = offset_row; j < scene->image_height; j += skip_row) {
		if (verbose) message("\r  rendering %03d/%03d        ", j, scene->image_height);
		for (auto i = 0; i < scene->image_width; i++) {
			// init accumulated color
			image->at(i, j) = zero3f;
			// grab proper random number generator
			auto rng = &rngs->at(i, j);
			// foreach sample
			for (auto jj : range(scene->image_samples)) {
				for (auto ii : range(scene->image_samples)) {
					// compute ray-camera parameters (u,v) for the pixel and the sample
					auto u = (i + (ii + rng->next_float()) / scene->image_samples) /
						scene->image_width;
					auto v = (j + (jj + rng->next_float()) / scene->image_samples) /
						scene->image_height;
					// compute camera ray
					auto ray = transform_ray(scene->camera->frame,
						ray3f(zero3f, normalize(vec3f((u - 0.5f)*scene->camera->width,
						(v - 0.5f)*scene->camera->height, -1))));
					// set pixel to the color raytraced with the ray
					image->at(i, j) += pathtrace_ray(scene, ray, rng, 0);
				}
			}
			// scale by the number of samples
			image->at(i, j) /= (scene->image_samples*scene->image_samples);
		}
	}
	if (verbose) message("\r  rendering done        \n");

}

// pathtrace an image with multithreading if necessary
image3f pathtrace(Scene* scene, bool multithread) {
	// allocate an image of the proper size
	auto image = image3f(scene->image_width, scene->image_height);

	// create a random number generator for each pixel
	auto rngs = RngImage(scene->image_width, scene->image_height);

	// if multitreaded
	if (multithread) {
		// get pointers
		auto image_ptr = &image;
		auto rngs_ptr = &rngs;
		// allocate threads and pathtrace in blocks
		auto threads = vector<thread>();
		auto nthreads = thread::hardware_concurrency();
		for (auto tid : range(nthreads)) threads.push_back(thread([=](){
			return pathtrace(scene, image_ptr, rngs_ptr, tid, nthreads, tid == 0); }));
				for (auto& thread : threads) thread.join();
	}
	else {
		// pathtrace all rows
		pathtrace(scene, &image, &rngs, 0, 1, true);
	}

	// done
	return image;
}

// runs the raytrace over all tests and saves the corresponding images
int main(int argc, char** argv) {
	auto args = parse_cmdline(argc, argv,
	{ "05_pathtrace", "raytrace a scene",
	{ { "resolution", "r", "image resolution", "int", true, jsonvalue() } },
	{ { "scene_filename", "", "scene filename", "string", false, jsonvalue("scene.json") },
	{ "image_filename", "", "image filename", "string", true, jsonvalue("") } }
	});
	auto scene_filename = args.object_element("scene_filename").as_string();
	auto image_filename = (args.object_element("image_filename").as_string() != "") ?
		args.object_element("image_filename").as_string() :
		scene_filename.substr(0, scene_filename.size() - 5) + ".png";
	auto scene = load_json_scene(scene_filename);
	if (not args.object_element("resolution").is_null()) {
		scene->image_height = args.object_element("resolution").as_int();
		scene->image_width = scene->camera->width * scene->image_height / scene->camera->height;
	}
	accelerate(scene);
	message("rendering %s ... ", scene_filename.c_str());
	auto image = pathtrace(scene, true);
	write_png(image_filename, image, true);
	delete scene;
	message("done\n");
}
