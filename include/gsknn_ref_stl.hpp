/*
 * --------------------------------------------------------------------------
 * GSKNN (General Stride K-Nearest Neighbors)
 * --------------------------------------------------------------------------
 * Copyright (C) 2015, The University of Texas at Austin
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *
 * gsknn_ref_stl.hpp
 *
 * Chenhan D. Yu - Department of Computer Science,
 *                 The University of Texas at Austin
 *
 *
 * Purpose:
 * this header file contains STL reference functions prototypes.
 *
 * Todo:
 *
 *
 * Modification:
 *
 * 
 * */



void dgsknn_ref_stl(
    int    m,
    int    n,
    int    k,
    int    r,
    double *XA,
    double *XA2,
    int    *alpha,
    double *XB,
    double *XB2,
    int    *beta,
    double *D,
    int    *I
    );

void sgsknn_ref_stl(
    int    m,
    int    n,
    int    k,
    int    r,
    float  *XA,
    float  *XA2,
    int    *alpha,
    float  *XB,
    float  *XB2,
    int    *beta,
    float  *D,
    int    *I
    );
