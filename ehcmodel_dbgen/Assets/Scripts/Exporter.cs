/* Author: András Sárkány and Zoltán Tősér
 License: BSD 3-clause
 Copyright (c) 2017, ELTE
 */
 
﻿using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/* Class for creating the output, exporting visible cube data 
 * from all positions, all directions
 */
public class Exporter
{
	private const string output_dir = "output/";

	private IList<GameObject> cubes;
	private bool occlusion_for_visibility_check;
	private bool plot_debug_raycast_spheres;
	private Camera main_camera;
	private StreamWriter camera_output;
	private StreamWriter cube_visibility_output;

	public Exporter (IList<GameObject> cubes, bool occlusion_for_visibility_check,
	                 bool plot_debug_raycast_spheres)
	{
		this.cubes = cubes;
		this.occlusion_for_visibility_check = occlusion_for_visibility_check;
		this.plot_debug_raycast_spheres = plot_debug_raycast_spheres;
		main_camera = GameObject.Find ("Main Camera").GetComponent<Camera> ();

		if (!Directory.Exists (output_dir)) {
			Directory.CreateDirectory (output_dir);
		}

		camera_output = new StreamWriter (output_dir + "camera_positions_and_orientations.txt");
		cube_visibility_output = new StreamWriter (output_dir + "cube_visibilities.txt");
	}

	public void ExportCubePositionsAndColors ()
	{
		StreamWriter cube_positions = new StreamWriter (output_dir + "cube_positions_and_colors.txt");
		foreach (GameObject cube in cubes) {
			Vector3 pos = cube.transform.position;
			Color color = cube.GetComponent<Renderer> ().material.color;
			cube_positions.WriteLine (pos.x + " " + pos.y + " " + pos.z + " " +
				color.r + " " + color.g + " " + color.b);
		}
		cube_positions.Flush ();
	}

	public void ExportSnapshot ()
	{
		Vector3 pos = main_camera.transform.position;
		Vector3 rot = main_camera.transform.eulerAngles;
		camera_output.WriteLine (pos.x + " " + pos.y + " " + pos.z + " " +
			rot.x + " " + rot.y + " " + rot.z);

		string visibilities = string.Join (" ",
		                                   GetVisibilityOfCubes ().ToList ().Select (x => x.ToString ()).ToArray ());
		//Debug.Log (visibilities);
		cube_visibility_output.WriteLine (visibilities);
	}

	public void ExportTopCamera()
	{
		Application.CaptureScreenshot (output_dir + "boxpos.png");
	}

	public void Flush ()
	{
		camera_output.Flush ();
		cube_visibility_output.Flush ();
	}

	public void Screenshot (int index)
	{
		Application.CaptureScreenshot (output_dir + "screenshot_" + index + ".png");
	}

	private IList<int> GetVisibilityOfCubes ()
	{
		IList<int> ret = new List<int> ();
		foreach (GameObject cube in cubes) {
			Plane[] frustum_planes = GeometryUtility.CalculateFrustumPlanes (main_camera);
			Bounds bounds = cube.GetComponent<Renderer> ().bounds;
			if (GeometryUtility.TestPlanesAABB (frustum_planes, bounds)) {
				if (!occlusion_for_visibility_check) {
					ret.Add (1);
				} else if (IsCubeOccluded (cube,10)) {
					ret.Add (0);
				} else {
					ret.Add (1);
				}
			} else {
				ret.Add (0);
			}
		}
		return ret;
	}

	/* Check the visibility of a single cube by raycasting at 
	 * the 8 vertices of the cube. If none of them can be seen then
	 * the cube is occluded
	 */
	private bool IsCubeOccluded (GameObject cube,int edge_points_no = 5)
	{
		bool ret = true;
		Mesh mesh = cube.GetComponent<MeshFilter> ().sharedMesh;
		GameObject raycastspheres = GameObject.Find ("RaycastSpheres");
		int[] v1ids = new int[]{0,0,1,2,1,3,0,2,7,4,4,5};
		int[] v2ids = new int[]{1,2,3,3,7,5,6,4,6,5,6,7};
		for (int i = 0; i< v1ids.Length;++i){
			Vector3 vertex1_in_local_space;
			Vector3 vertex2_in_local_space;
			vertex1_in_local_space = mesh.vertices[v1ids[i]];
			vertex2_in_local_space = mesh.vertices[v2ids[i]];

			Vector3 vertex1 = cube.transform.TransformPoint (vertex1_in_local_space);
			Vector3 vertex2 = cube.transform.TransformPoint (vertex2_in_local_space);

			Vector3 edge_vector = vertex2-vertex1;
			for (int j=0;j<=edge_points_no;++j) {

				Vector3 vertex_on_edge = vertex1+(1.0f/edge_points_no)*j*edge_vector;
				if (plot_debug_raycast_spheres && raycastspheres.transform.childCount<1000){
					GameObject sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
					sphere.transform.parent = raycastspheres.transform;
					sphere.transform.position = vertex_on_edge;
					sphere.transform.localScale = new Vector3(0.05f,0.05f,0.05f);
					sphere.GetComponent<Renderer> ().material.color = new Color(1.0f,0f,0f);
				}	
				RaycastHit hitinfo;
				Vector3 camera_pos = main_camera.transform.position;

				// Emit a ray from the camera towards the vertex.
				// If we hit the cube of the vertex, then it is not occluded.
				// If we hit another cube, then THIS VERTEX is occluded (we should check the other though).
				// If we hit nothing, then it is not occluded (maybe this occurs in case of a collision
				// detection in an extremal situation).
				bool hit = Physics.Raycast (camera_pos, vertex_on_edge - camera_pos, out hitinfo);
				if (!hit || hitinfo.collider.gameObject == cube) {
					ret = false;
					break;
				}
			}
			if (!ret) break;
		}
		return ret;
	}
}
