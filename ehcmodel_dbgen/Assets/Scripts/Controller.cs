/* Author: András Sárkány and Zoltán Tősér
 License: BSD 3-clause
 Copyright (c) 2017, ELTE
 */
﻿using UnityEngine;
using System.Collections;
using System.Collections.Generic;

/*This is the main script, it should be attached to 
 * the "Main camera" Gameobject
*/
public class Controller : MonoBehaviour
{
	private IList<GameObject> cubes;
	private Camera main_camera;
	private Exporter exp;

	private float room_size = 15f; // Size of the room
	private float step_size = 0.1f; //Size of a single step

	private void Start ()
	{
		Random.seed = 42;
		main_camera = GameObject.Find ("Main Camera").GetComponent<Camera> ();
		//Setting the field of view
		main_camera.fieldOfView = 20 * (3f / 4f);// / ((float)main_camera.pixelWidth / main_camera.pixelHeight);
		//cubes = CubeFactory.GenerateCubes ();
		CubeFactory cf = new FreeSpaceCubeFactory (150,15f+0.5f,17f);
		//CubeFactory cf = new CubeFactory (30,5f);
		//CubeFactory cf = new HardcodedCubeFactory (room_size,"test2");
		//CubeFactory cf = new TriangleCubeFactory(150,15f+1f,1.5f);
		cubes = cf.GenerateCubes ();
		exp = new Exporter (cubes, true,false);
		exp.ExportCubePositionsAndColors ();
		Camera top_camera = GameObject.Find ("Top Camera").GetComponent<Camera> ();
		/*if (top_camera.rect.x < 0.00001 && top_camera.rect.y < 0.00001) {
			GameObject.Find ("Camera Triangle").SetActive(false);
			exp.ExportTopCamera ();
		}*/

		//StartCoroutine(camera_ready());
		StartCoroutine (UpdateState ());
	}

	/*The main loop going through all the positions and directions
	 */
	private IEnumerator UpdateState ()
	{	



		//top_camera.rect = new Rect (0.7f, 0.7f, 1f, 1f);

		int data_counter = 0;
		for (float pos_x = -room_size/2; pos_x <= room_size/2; pos_x += step_size) {
			System.DateTime starttime = System.DateTime.Now;
			for (float pos_z = -room_size/2; pos_z <= room_size/2; pos_z += step_size) {
				for (float rot_y = 0f; rot_y < 360; rot_y += 10) {
					main_camera.transform.position = new Vector3 (pos_x, 0, pos_z);
					main_camera.transform.rotation = Quaternion.Euler (0, rot_y, 0);
					
					if (!CameraCollidesWithCubes ()) {
						exp.ExportSnapshot ();
						if (Random.value < 0.001) {
							exp.Screenshot (data_counter);
						}
						++data_counter;
					}

					yield return null;
				}
			}
			System.DateTime endtime = System.DateTime.Now;
			System.TimeSpan timeDifference = endtime.Subtract(starttime);
			Debug.Log(timeDifference.Seconds);
			Debug.Log(timeDifference.Seconds*room_size/step_size);
		}
		
		exp.Flush ();
		Debug.Log ("Done");
	}

	private bool CameraCollidesWithCubes ()
	{
		bool ret = false;
		foreach (GameObject cube in cubes) {
			if (cube.GetComponent<Renderer> ().bounds.Contains (main_camera.transform.position)) {
				ret = true;
				break;
			}
		}
		return ret;
	}

	private IEnumerator camera_ready() {
		for (int i = 0; i<3; ++i) {
			yield return null;
			Debug.Log (i);
			main_camera.transform.position = new Vector3 (-6, 0, -4);
			main_camera.transform.rotation = Quaternion.Euler (0, 200 + i * 20, 0);
			exp.Screenshot (i);
		}

		Debug.Log ("CR Done");
	}
}
