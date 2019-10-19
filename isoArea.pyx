def compute_isoArea(self,c_iso):
    print('Computing the surface for c_iso: ', c_iso)

    half_filter = int(self.filter_width/2)

    # reference area of planar flame
    A_planar = (self.filter_width - 1)**2

    isoArea_coefficient = np.zeros((self.Nx,self.Nx,self.Nx))
    for l in range(half_filter, self.Nx - half_filter):
        for m in range(half_filter, self.Nx - half_filter):
            for n in range(half_filter, self.Nx - half_filter):
                this_LES_box = (self.c_data_np[l-half_filter : l+half_filter,
                                                  m-half_filter : m+half_filter,
                                                  n-half_filter : n+half_filter])

                # this works only if the c_iso value is contained in my array
                # -> check if array contains values above AND below iso value
                if np.any(np.where(this_LES_box < c_iso)) and np.any(np.any(np.where(this_LES_box > c_iso))):
                    verts, faces = measure.marching_cubes_classic(this_LES_box, c_iso)
                    iso_area = measure.mesh_surface_area(verts=verts, faces=faces)
                else:
                    iso_area = 0

                isoArea_coefficient[l,m,n] = iso_area / A_planar

    return isoArea_coefficient