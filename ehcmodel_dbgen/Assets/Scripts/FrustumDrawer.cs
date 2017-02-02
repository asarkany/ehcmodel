/* Author: András Sárkány and Zoltán Tősér
 License: BSD 3-clause
 Copyright (c) 2017, ELTE
 */
﻿using UnityEngine;
using System.Collections;

/* Class for drawing the field of view on the top view */
public class FrustumDrawer : MonoBehaviour
{
	private Camera main_camera;
	private MeshFilter mesh_filter;

	private void Start ()
	{
		main_camera = GameObject.Find ("Main Camera").GetComponent<Camera> ();
		mesh_filter = gameObject.GetComponentInChildren<MeshFilter> ();
	}
	
	private void LateUpdate ()
	{

		Plane[] planes = GeometryUtility.CalculateFrustumPlanes (main_camera);
		GameObject p;
		/*
		if (main_camera.transform.position.x != 0.0 && !GameObject.Find ("Plane 0")) {
			for (int i = 0; i < 2; i++) {
				p = GameObject.CreatePrimitive (PrimitiveType.Plane);
				p.name = "Plane " + i.ToString ();
				p.transform.position = -planes [i].normal * planes [i].distance;
				p.transform.rotation = Quaternion.FromToRotation (Vector3.up, planes [i].normal);
			}
		}
		*/

		Plane left_plane = planes [0];
		Plane right_plane = planes [1];

		Vector3 point1 = main_camera.transform.position;
		Vector3 center = point1 + 50 * main_camera.transform.forward;
		Vector3 point2 = Vector3.ProjectOnPlane (center, left_plane.normal);
		Vector3 point3 = Vector3.ProjectOnPlane (center, right_plane.normal);

		/*Vector3 point1i = transform.InverseTransformPoint (point1);
		Vector3 point2i = transform.InverseTransformPoint (point2);
		Vector3 point3i = transform.InverseTransformPoint (point3);
		*/
		/*if (main_camera.transform.position.x != 0.0 && !GameObject.Find ("Sphere 0")) {
			p = GameObject.CreatePrimitive (PrimitiveType.Sphere);
			p.transform.position = point1;
			p.name = "Sphere 0";
			p = GameObject.CreatePrimitive (PrimitiveType.Sphere);
			p.transform.position = point2;
			p = GameObject.CreatePrimitive (PrimitiveType.Sphere);
			p.transform.position = point3;
			p = GameObject.CreatePrimitive (PrimitiveType.Sphere);
			p.transform.position = center;
			*/
			/*
			p = GameObject.CreatePrimitive (PrimitiveType.Sphere);
			p.transform.position = point1i;
			p = GameObject.CreatePrimitive (PrimitiveType.Sphere);
			p.transform.position = point2i;
			p = GameObject.CreatePrimitive (PrimitiveType.Sphere);
			p.transform.position = point3i;
			*/
			/*
			Debug.Log (point1);
			Debug.Log (point2);
			Debug.Log (point3);
			Debug.Log (center);
			Debug.Log (transform.gameObject);
			Debug.Log (main_camera.transform.forward);
		}
		*/

		point1 = transform.InverseTransformPoint (point1);
		point2 = transform.InverseTransformPoint (point2);
		point3 = transform.InverseTransformPoint (point3);



		Vector3[] vertices = {point1, point2, point3};
		mesh_filter.mesh.vertices = vertices;
		mesh_filter.mesh.triangles = new int[] {0, 1, 2};
		mesh_filter.mesh.RecalculateNormals ();
	}
}
