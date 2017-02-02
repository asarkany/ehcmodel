/* Author: Zoltán Tősér
 License: BSD 3-clause
 Copyright (c) 2017, ELTE
 */
 ﻿using UnityEngine;
using System.Collections;
using System.Collections.Generic;

/* Simple class for generating a square with boxes (placed randomly) in it
 */
public class CubeFactory//:AbstractCubeFactory
{
	protected int cube_num = 30;
	protected float room_size = 5;

	public CubeFactory(int cube_num,float room_size)
	{
		this.cube_num = cube_num;
		this.room_size = room_size;
	}

	virtual public IList<GameObject> GenerateCubes ()
	{
		GameObject cube_container = GameObject.Find ("Cubes");
		GameObject original_cube = Resources.Load<GameObject> ("Prefabs/Cube");
		IList<GameObject> cubes = new List<GameObject> ();

		for (int i = 0; i < cube_num; ++i) {
			GameObject cube = GameObject.Instantiate (original_cube);
			cube.transform.parent = cube_container.transform;
			cube.name = "Cube" + i;
			GiveRandomColor (cube);
			PlaceNewCube (cube, cubes);
			cubes.Add (cube);
		}

		return cubes;
	}

	protected void GiveRandomColor (GameObject cube)
	{
		Material material = cube.GetComponent<Renderer> ().material;
		float r = Random.value;
		float g = Random.value;
		float b = Random.value;
		material.color = new Color (r, g, b);
	}

	virtual protected void PlaceNewCube (GameObject cube, IList<GameObject> existing_cubes)
	{
		do {
			float new_x = room_size * Random.value - room_size / 2;
			float new_z = room_size * Random.value - room_size / 2;
			cube.transform.position = new Vector3 (new_x, 0, new_z);
		} while (this.CollidesWithExistingCubes(cube, existing_cubes));
	}

	protected bool CollidesWithExistingCubes (GameObject new_cube, IList<GameObject> existing_cubes)
	{
		bool ret = false;
		foreach (GameObject cube in existing_cubes) {
			if (cube.GetComponent<Collider> ().bounds.Intersects (new_cube.GetComponent<Collider> ().bounds)) {
				ret = true;
				break;
			}
		}
		return ret;
	}
}
