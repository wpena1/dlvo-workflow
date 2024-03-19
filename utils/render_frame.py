import PIL
import fresnel
import gsd.hoomd
import numpy as np

def render_frame_gsd(frame_number, output_prefix):
    cpu = fresnel.Device(mode='cpu')
    frame_number = int(frame_number)
    output_prefix = str(output_prefix)
    full_path = output_prefix+".gsd"
    with gsd.hoomd.open(name=F"{full_path}", mode='r') as gsd_file:
        snap = gsd_file[frame_number]

    box = snap.configuration.box
    N = snap.particles.N
    particle_types = snap.particles.typeid
    colors = np.empty((N, 3))


    # Color by typeid
    colors[particle_types == 0] = fresnel.color.linear([0.9, 0, 0]) # A type
    colors[particle_types == 1] = fresnel.color.linear([0, 0.9, 0]) # B type
    scene = fresnel.Scene(device=cpu)
    # scene.lights = fresnel.light.butterfly()
    scene.background_alpha = 1.0
    scene.background_color = fresnel.color.linear([1, 1, 1])

    # Spheres for every particle in the system
    geometry = fresnel.geometry.Sphere(scene, N=N)
    geometry.position[:] = snap.particles.position
    geometry.radius[:] = (snap.particles.diameter)/2.0
    geometry.material = fresnel.material.Material(solid=0.05, roughness=0.2, specular=0.8)
    geometry.outline_width = 15

    # use color instead of material.color
    geometry.material.primitive_color_mix = 1.0
    geometry.color[:] = fresnel.color.linear(colors)

    fresnel.geometry.Box(scene, box, box_radius=20, box_color=fresnel.color.linear([0, 0, 0]))

    scene.camera = fresnel.camera.Orthographic.fit(scene, view='front')
    scene.lights = fresnel.light.lightbox()
    temp_img=fresnel.pathtrace(scene, samples=256, light_samples=64, w=1500, h=1500)
    img = PIL.Image.fromarray(temp_img[:], mode='RGBA')
    img.save(F"{output_prefix}frame_{frame_number}.png")
    return 1


if __name__=="__main__":

    import sys
    frame_num = [0, -1]
    outputprefix = sys.argv[1]
    for f in frame_num:
        r = render_frame_gsd(f ,outputprefix)

