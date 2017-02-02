/* Author: András Sárkány
 License: BSD 3-clause
 Copyright (c) 2017, ELTE
 */
﻿using UnityEngine;
using System.Collections;
using System.Collections.Generic;

/* Special cubefactory for generating randomly placed cubes in
 * a square but leaving space in the inside in the shape of a smaller square
 */
public class FreeSpaceCubeFactory : CubeFactory {

	protected float outer_ring_size = 4;

	public FreeSpaceCubeFactory(int cube_num,float room_size, float outer_ring_size):base(cube_num,room_size)
	{
		this.outer_ring_size = outer_ring_size;
	}

	override protected void PlaceNewCube (GameObject cube, IList<GameObject> existing_cubes)
	{
		do {
			//float new_x = ((outer_ring_size-room_size)/2 * Random.value + room_size)*System.Math.Sign(Random.value-0.5);
			//float new_z = ((outer_ring_size-room_size)/2 * Random.value + room_size)*System.Math.Sign(Random.value-0.5);
			float new_x = outer_ring_size * Random.value - outer_ring_size / 2;
			float new_z = outer_ring_size * Random.value - outer_ring_size / 2;
			cube.transform.position = new Vector3 (new_x, 0, new_z);
		} while (InInnerRing(cube) || CollidesWithExistingCubes(cube, existing_cubes));
	}


	protected bool InInnerRing (GameObject cube)
	{
		return System.Math.Abs(cube.transform.position.x)-room_size / 2<0 && 
			System.Math.Abs(cube.transform.position.z)-room_size / 2<0;
	}
}
