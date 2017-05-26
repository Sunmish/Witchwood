# Witchwood --------------------------------------------------------------------
#
# Simple source-finding and flux-measuring.
# Either get measurements for all sources in an image, or do source-finding and
# cross-reference the returned catalogue to a specific source.
# This makes heavy use of Astropy.
#
# Tested to work with the following survey stamps:
#   NVSS
#   TGSS
#   VLSSr
#   GLEAM
# 
# Exciting new features!
# - Measure flux in a circular aperture!
# - Measure flux in a polygon defined by some number of vertices! (N_vertices > 3)
#
#
# TODO:
#    General speed improvement would be good. Not sure where to start with this.
#    Need to test on SUMSS postage stamps.
# ------------------------------------------------------------------------------


import numpy
import math
import os
import sys

from datetime import datetime             # For timing purposes.
from astropy.io import fits               # For handling FITS files.
from astropy.wcs import WCS               # For computing LAS/positions.
from astropy.coordinates import SkyCoord  # For cross-referencing.
from astropy import units as u            # For cross-referencing.


def angular_distance(coords1, coords2):
    '''Get the angular distance between a set of RA, DEC coordinates in [dd].'''

    cos_angle = math.sin(math.radians(coords1[1])) * \
            math.sin(math.radians(coords2[1])) + \
            math.cos(math.radians(coords1[1])) * \
            math.cos(math.radians(coords2[1])) * \
            math.cos(math.radians(coords1[0] - coords2[0]))

    try:
        gamma = math.degrees(math.acos(cos_angle))
    except ValueError:
        gamma = math.degrees(math.acos(min(1, max(cos_angle, -1))))

    return gamma


def read_fits(fitsfile=None, hdulist=None):
    '''Read a FITS file and return relevant data.

    Because the FITS standard is in fact not completely standard there a few 
    little things that need to be checked to looked for the necessary header 
    data.
    '''

    if fitsfile is not None:
        if not fitsfile.endswith('.fits'):
            fitsfile += '.fits'
        hdulist = fits.open(fitsfile)

    elif hdulist is not None:
        hdulist = hdulist

    else:
        raise RuntimeError('To read a FITS file you must at least provide a '\
            'FITS file.')

    harray = hdulist[0].header
    farray = hdulist[0].data
    warray = WCS(hdulist[0].header)

    # Read in some things from the FITS header:
    # Check dimensions - must only have two axes:
    try:
        farray.shape = (harray['NAXIS2'], harray['NAXIS1'])
    except ValueError:
        raise ValueError('FITS file must be flat.')
    # Find pixel sizes:
    try:
        cd1 = harray['CDELT1']     # Normal key for pixel size.
        cd2 = harray['CDELT2']
    except KeyError:
        try:
            cd1 = harray['CD1_1']  # Abnormal key (e.g., GLEAM)
            cd2 = harray['CD2_2']
        except KeyError:
            raise KeyError('Cannot find key for pixel sizes.')
    # Beam parameters:
    semi_a, semi_b, beam_area = None, None, None
    try:
        semi_a = 0.5 * harray['BMAJ']  # Normal beam keys.
        semi_b = 0.5 * harray['BMIN']
    except KeyError:
        try:
            semi_a = 0.5 * harray['CLEANBMJ']  # Abnormal beam keys (e.g., VLSSr)
            semi_b = 0.5 * harray['CLEANBMN']
        except KeyError:
            try:  # Check for NVSS, which has the same beam information for all stamps.
                for i in range(0, len(hdulist[0].header['HISTORY'])):
                    if 'NRAO VLA SKY SURVEY' in hdulist[0].header['HISTORY'][i]:
                        semi_a = 0.5 * 1.2500E-02
                        semi_b = 0.5 * 1.2500E-02
                if semi_a is None:
                    raise KeyError
            except KeyError:
                try:  # AIPS has a tell-tale mark:
                    for i in range(0, len(hdulist[0].header['HISTORY'])):
                        if 'BMAJ' in hdulist[0].header['HISTORY'][i]:
                            l = []
                            for t in hdulist[0].header['HISTORY'][i].split():
                                try:
                                    l.append(float(t))
                                except ValueError:
                                    pass
                            semi_a = 0.5 * l[0]
                            semi_b = 0.5 * l[1]
                    if semi_a is None:
                        raise KeyError
                except (KeyError, IndexError):
                    try:  # Check for SUMSS, which does NOT have the beam information.
                        for i in range(0, len(hdulist[0].header['HISTORY'])):
                            if 'Sydney University Molonglo Sky Survey (SUMSS)' \
                                in hdulist[0].header['HISTORY'][i]:
                                decl = input('Declination of source: ')
                                decl = numpy.radians(abs(decl))
                                beam_area = numpy.pi * (45.0**2 / 3600.0**2) * \
                                    (1.0 / math.sin(decl))
                        if beam_area is None:
                            raise KeyError
                    except KeyError:
                        beam_params = input('BEAM size (a, b) [deg]: ')
                        semi_a = 0.5 * beam_params[0]
                        semi_b = 0.5 * beam_params[1]
                    # raise KeyError('Cannot find key for beam parameters.')

    if beam_area is None:
        beam_area = (numpy.pi * semi_a * semi_b)
    beams_per_pixel = beam_area / (abs(cd1*cd2) * numpy.log(2))
    
    return farray, warray, beams_per_pixel, cd1, cd2, hdulist


class Tree():
    '''Grow a tree with branches and leaves.

    Grow a source from neighbouring pixels.
    '''


    def __init__(self, tree_number, threshold):
        '''"The seed of a tree of giants."'''
        self.no = tree_number     # The source ID.
        self.th = threshold[0]    # Threshold needed for detection.
        self.gh = threshold[1]    # Growing threshold.

    def seedling(self, m, n, farray, warray, forest, diagonals):
        '''Start from a seedling and grow a tree.'''

        if farray[m, n] >= self.th:                    # Detection!
            forest[m, n] = self.no                     # Set ref to ID
            self.leaves = 1                            # Count pixels
            self.fluxes = numpy.array([farray[m, n]])  # Add flux
            self.coords = numpy.array([(m, n)])        # Add pixel coordinates
            self.bounds = numpy.array([(m, n)])        # Boundary coordinates
            self.bright = farray[m, n]                 # Brightest pixel
            self.bcoord = [(m, n)]                     # BP coordinates

            indices = [(m, n)]  # Indices to check. This is added to.

            for i, j in indices:
                # Surrounding pixels:
                if diagonals:
                    surrounding_indices = [(i-1, j-1), (i-1, j), (i, j-1), \
                        (i+1, j-1), (i-1, j+1), (i+1, j), (i, j+1), (i+1, j+1)]
                else:
                    surrounding_indices = [(i-1, j), (i, j-1), (i+1, j), \
                        (i, j+1)]

                boundary = False

                for index in surrounding_indices:
                    if (index[0] < 0) or (index[1] < 0):
                        pass
                    else:
                        try:
                            if (numpy.isnan(forest[index])) and \
                                (farray[index] >= self.gh):
                                self.leaves += 1
                                self.fluxes = numpy.append(self.fluxes, \
                                    farray[index])
                                self.coords = numpy.append(self.coords, [index], \
                                    axis=0)
                                forest[index] = self.no
                                if farray[index] > self.bright:
                                    self.bright = farray[index]
                                    self.bcoord = [index]
                                indices.append(index)
                            elif numpy.isnan(forest[index]):
                                forest[index] = 0
                                farray[index] = numpy.nan
                                if not boundary:
                                    self.bounds = numpy.append(self.bounds, \
                                        [index], axis=0)
                                    boundary = True
                            else:
                                if not boundary:
                                    self.bounds = numpy.append(self.bounds, \
                                        [index], axis=0)
                                    boundary = True
                        except IndexError:
                            pass

            tree_number = self.no + 1

        else:
            tree_number = self.no
            forest[m, n] = 0
            farray[m, n] = numpy.nan


        return farray, forest, tree_number


def populate_forest(farray, warray, threshold, max_pix, min_pix, diagonals):
    '''Grow trees in a forest; find sources in a field.'''

    # An empty forest:
    forest = numpy.empty((len(farray[:, 0]), len(farray[0, :])))
    forest[:] = numpy.nan

    tree_number = 1                 # The current source ID,
    tree_leaves = {tree_number: 0}  # its pixels,
    tree_fluxes = {tree_number: 0}  # its flux values,
    tree_coords = {tree_number: 0}  # its pixel coordinates,
    tree_bounds = {tree_number: 0}  # its boundary coordinates,
    tree_bright = {tree_number: 0}  # Source brightest pixel coordinates.
    tree_height = {tree_number: 0}

    for n in range(0, len(farray[0, :])):
        for m in range(0, len(farray[:, 0])):

            # If forest[m, n] is not NaN, it has already been checked.
            if numpy.isnan(forest[m, n]):
                t = Tree(tree_number, threshold)

                farray, forest, tree_number = t.seedling(m, n, farray, \
                    warray, forest, diagonals)

                try:
                    if (min_pix <= t.leaves <= max_pix):
                        tree_leaves[tree_number-1] = t.leaves
                        tree_fluxes[tree_number-1] = t.fluxes
                        tree_coords[tree_number-1] = t.coords
                        tree_bounds[tree_number-1] = t.bounds
                        tree_bright[tree_number-1] = t.bcoord
                        tree_height[tree_number-1] = t.bright

                    else:
                        pass
                except AttributeError:
                    pass
    return farray, forest, tree_leaves, tree_fluxes, tree_coords, tree_bounds, \
        tree_bright, tree_height


def measure_forest(fitsfile=None, hdulist=None, rms=None, cutoff1=None, cutoff2=None, \
    max_pix=500, min_pix=2, diagonals=True, LAS=True, output=None, annotations='ds9', \
    verbose=True):
    '''Calculate the fluxes of individual sources within a FITS file.

    Parameters
    ----------
    fits        : string
                FITS file and filepath.
    hdulist     : HDUList object, optional
                Can be specified in place of a FITS file.
    rms         : float
                rms of the image. Minimum detection threshold is `rms` * `cutoff`.
    cutoff1     : int, optional
                The multiple of `rms` needed for a detection. Default is 3sigma.
    cutoff2     : int, optional
                The multiple of `rms` needed for growing sources. Default is `cutoff1`.
    max_pix     : int, optional
                Maximum number of pixels in a detection. Useful only to save
                time ignoring large sources (e.g., Fornax A) as LAS calculations
                are ridiculously slow.
    min_pix     : int, optional
                Minimum number of pixels for a detection.
    diagonals   : bool, optional
                Specifies whether source detection counts pixels only connected
                diagonally. Select True for diagonal detection.
    LAS         : bool, optional
                Select True is wanting to calculate the largest angular size of
                each source. THIS IS VERY SLOW.
    output      : string, optional
                Output filename for both FITS file ouput of detections and a text
                file with sources and measured parameters. No extension.
    annotations : {'ds9', 'kvis'}, optional
                An annotation file is written if `output` is True. This specifies
                whether the annotations should be ds9 or kvis format.
    verbose     : bool, optional
                Select True if wanting to print output to terminal.
    '''

    if rms is None:
        raise ValueError('RMS must be specified.')
    if cutoff1 is None:
        cutoff1 = 3
    if cutoff2 is None:
        cutoff2 = cutoff1

    threshold1 = cutoff1 * rms
    threshold2 = cutoff2 * rms

    if output is not None:
        start_time = datetime.now()

    if fitsfile is not None:
        farray, warray, beams_per_pixel, cd1, cd2, hdulist = \
            read_fits(fitsfile=fitsfile)
    elif hdulist is not None:
        farray, warray, beams_per_pixel, cd1, cd2, hdulist = \
            read_fits(hdulist=hdulist)
    else:
        raise RuntimeError('Either a FITS file or HDUList object must be '\
            'specified.')

    naxis = hdulist[0].header['NAXIS']

    farray, forest, tree_leaves, tree_fluxes, tree_coords, tree_bounds,\
        tree_bright, tree_height = populate_forest(farray=farray, warray=warray, \
        threshold=(threshold1, threshold2), max_pix=max_pix, min_pix=min_pix, \
        diagonals=diagonals)

    source_flux = []
    source_dflux = []
    source_peak = []
    source_centroid = []
    source_avg_flux = []
    source_bcoord = []
    source_area = []
    source_npix = []
    source_LAS = []
    source = []


    if (len(tree_leaves) == 1) and (tree_leaves[1] == 0):
        raise ValueError('No sources detected for {0} with threshold = '\
                '{1}'.format(fitsfile, threshold1))

    zero_flag = False
    if tree_leaves[1] == 0:
        zero_flag = True
        del tree_leaves[1]
        del tree_fluxes[1]
        del tree_coords[1]
        del tree_bounds[1]
        del tree_bright[1]
        del tree_height[1]

    for tree in tree_leaves:

        try:
            source_flux.append(sum(tree_fluxes[tree]) / beams_per_pixel)
            source_dflux.append(rms * numpy.sqrt(float(tree_leaves[tree]) / \
                beams_per_pixel))
            source_avg_flux.append(sum(tree_fluxes[tree]) / tree_leaves[tree])
            source_area.append(abs(cd1*cd2) * tree_leaves[tree])
            source_npix.append(tree_leaves[tree])
            source_peak.append(tree_height[tree])
            try:
                source_bcoord.append((tree_bright[tree][0][1], \
                    tree_bright[tree][0][0]))
            except IndexError:
                pass
        except TypeError:
            # raise ValueError('Threshold may be set as 0. This causes problems.')
            raise

        # Source centroid. The mean position weighted to the flux/beam.
        fx = []
        fy = []
        fz = sum(tree_fluxes[tree])
        for i in range(0, len(tree_fluxes[tree])):
            fx.append(tree_coords[tree][i][0] * tree_fluxes[tree][i])
            fy.append(tree_coords[tree][i][1] * tree_fluxes[tree][i])
        # Note the coordinate order is (y, x).
        source_centroid.append((sum(fy) / fz, sum(fx) / fz))

        # The largest angular scale of the source. Simply the largest separation
        # between any two pixels. This is optional as it takes significantly
        # longer than the rest of the script.
        # TODO: fix NAXS > 2 issue like below. 
        if LAS and (naxis == 2):
            length = 0
            for pix_coord1 in range(0, len(tree_bounds[tree])):
                ra1, dec1 = \
                    warray.all_pix2world(tree_bounds[tree][pix_coord1][1], \
                    tree_bounds[tree][pix_coord1][0], 0)
                for pix_coord2 in range(pix_coord1+1, len(tree_bounds[tree])):
                    ra2, dec2 = \
                        warray.all_pix2world(tree_bounds[tree][pix_coord2][1], \
                        tree_bounds[tree][pix_coord2][0], 0)
                    if (ra1 == ra2) and (dec1 == dec2):
                        diff = 0.0
                    else:
                        diff = angular_distance((ra1, dec1), (ra2, dec2))
                    if diff > length:
                        length = diff
            source_LAS.append(length)
        else:
            source_LAS.append('NA')

        if zero_flag:              # Account for empty "zeroth" sources from
            source.append(tree-1)  # intialising the source dict.
        else:
            source.append(tree)

    # Converting coordinates depends on number of axes of the FITS file.
    try:
        if naxis == 2:
        # try:
            world_coords = warray.all_pix2world(source_centroid, 0)
            bright_coords = warray.all_pix2world(source_bcoord, 0)
        else:
        # except ValueError:
            wx, wy, bx, by = [], [], [], []
            for i in range(0, len(source)):
                wx.append(source_centroid[i][0])
                wy.append(source_centroid[i][1])
                bx.append(source_bcoord[i][0])
                by.append(source_bcoord[i][1])

            if naxis == 3:
            # try:
                wc = warray.all_pix2world(wx, wy, numpy.ones_like(wx), 0)
                bc = warray.all_pix2world(bx, by, numpy.ones_like(bx), 0)
            # except ValueError:
                # try:
            elif naxis == 4:
                    wc = warray.all_pix2world(wx, wy, numpy.ones_like(wx), numpy.ones_like(wx), 0)
                    bc = warray.all_pix2world(bx, by, numpy.ones_like(bx), numpy.ones_like(bx), 0)
            else:
                # except ValueError:
                raise ValueError('NAXIS size must be 2, 3, or 4.')
            world_coords = []
            bright_coords = []
            for i in range(0, len(source)):
                world_coords.append([wc[0][i], wc[1][i]])
                bright_coords.append([bc[0][i], bc[1][i]])

    except TypeError:
        # world_coords, bright_coords = [], []
        # for i in range(0, len(source)):
        #     world_coords.append(('NA', 'NA'))
        #     bright_coords.append(('NA', 'NA'))
        raise


    if output is not None:
        with open(output+'.txt', 'w+') as f:

            f.write('# Sources and their parameters measured by Witchwood.\n')
            f.write('# \n')
            f.write('# ............ Input FITS = {0}\n'.format(fitsfile))
            f.write('# ................ Output = {0}\n'.format(output))
            f.write('# ............. Threshold = {0}\n'.format(threshold1))
            f.write('# ........ Minimum pixels = {0}\n'.format(min_pix))
            f.write('# ........ Maximum pixels = {0}\n'.format(max_pix))
            f.write('# Total number of sources = {0}\n'.format(len(source)))
            f.write('# ............ Time taken = {0}\n'.format( \
                datetime.now()-start_time))
            f.write('# ................... LAS = {0}\n'.format(LAS))
            f.write('# \n')
            f.write('# source flux dflux avg_flux area npix centroid_RA centroid_DEC bright_RA bright_DEC LAS\n')
            f.write('#         Jy    Jy  Jy/Beam  deg^2         deg         deg          deg      deg     deg\n')
            f.write('#   0     1     2      3      4     5       6           7            8        9       10\n')

            for i in range(0, len(source)):
                f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}\n'.format(\
                        source[i], \
                        source_flux[i], source_dflux[i], source_avg_flux[i], \
                        source_area[i], source_npix[i], world_coords[i][0], \
                        world_coords[i][1], bright_coords[i][0], \
                        bright_coords[i][1], source_LAS[i]))

        if annotations in ['ds9', 'DS9']:
                with open(output+'.reg', 'w+') as reg:
                    reg.write('global color=yellow\n')
                    reg.write('fk5\n')

                    for i in range(0, len(source)):
                        reg.write('text {0} {1} '.format(bright_coords[i][0], \
                            bright_coords[i][1])+r'{'+'{0}'.format(source[i])+ \
                            r'}'+'\n')

        elif annotations in ['karma', 'Karma', 'KARMA', 'kvis', 'KVIS', 'Kvis']:
                with open(output+'.ann', 'w+') as ann:
                    ann.write('COLOR YELLOW\n')
                    ann.write('COORD W\n')

                    for i in range(0, len(source)):
                        ann.write('TEXT {0} {1} {2}\n'.format(world_coords[i][0], \
                            world_coords[i][1], source[i]))

        elif annotations is not None:
                raise ValueError('Annotations must be "ds9" or "kvis".')

        else:
            pass

        # Output the blanked FITS file.
        hdulist[0].data = numpy.array(farray)
        if os.path.exists(output+'.fits'):
            print 'Deleting old {0}.fits'.format(output)
            os.remove(output+'.fits')
        hdulist.writeto(output+'.fits')
        # Output a FITS file with the sources and their IDs.
        hdulist[0].data = numpy.array(forest, dtype=float)
        hdulist[0].header['BUNIT'] = 'Source number'
        if os.path.exists(output+'_forest.fits'):
            print 'Deleting old {0}_forest.fits'.format(output)
            os.remove(output+'_forest.fits')
        hdulist.writeto(output+'_forest.fits')

    return source, source_flux, source_dflux, source_avg_flux, source_area, \
        source_npix, world_coords, bright_coords, source_LAS, source_peak


def measure_tree(fitsfile, RA, DEC, rms, cutoff1=3, cutoff2=None, max_pix=500, min_pix=2, \
    diagonals=True, LAS=True, annotations='ds9', output=None, verbose=False):
    '''Measure the flux/length of a single source.

    Parameters
    ----------
    fits      : string
              FITS file and filepath.
    RA        : string or float
              Right ascension of source to be measured.
    DEC       : string or float
              Declination of source to be measured.
    rms       : float
              rms of the image. Minimum detection threshold is `rms` * `cutoff1`.
    cutoff1    : int, optional
              The multiple of `rms` needed for a detection. Default is 3sigma.
    max_pix   : int, optional
              Maximum number of pixels in a detection. Useful only to save
              time ignoring large sources (e.g., Fornax A) as LAS calculations
              are ridiculously slow.
    min_pix   : int, optional
              Minimum number of pixels for a detection.
    diagonals : bool, optional
              Specifies whether source detection counts pixels only connected
              diagonally. Select True for diagonal detection.
    LAS       : bool, optional
              Select True is wanting to calculate the largest angular size of
              each source. This can be very slow depending on sizes of arrays.
    output    : string, optional
              Specify a filename and path for output text, annotation, and FITS
              files.
    verbose   : bool, optional
              If True then results will be printed to the terminal/console.
    '''


    # A forest in which our tree resides.
    source, source_flux, source_dflux, source_avg_flux, source_area, \
        source_npix, world_coords, bright_coords, source_LAS, source_peak = \
        measure_forest(fitsfile=fitsfile, rms=rms, cutoff1=cutoff1, \
        cutoff2=cutoff2, max_pix=max_pix, min_pix=min_pix, diagonals=diagonals, LAS=LAS, \
        output=output, verbose=False, annotations=annotations)

    # Now to find the tree:
    c = SkyCoord(RA, DEC, unit=(u.deg, u.deg))
    ww_catalogue = SkyCoord(world_coords, unit=(u.deg, u.deg))
    i = c.match_to_catalog_sky(ww_catalogue)[0]

    # As a check to see if the found source is in fact the source in question.
    dist = angular_distance((RA, DEC), (world_coords[i][0], world_coords[i][1]))

    if verbose:
        print '                Int. flux = {0} [Jy]'.format(source_flux[i])
        print '          Error int. flux = {0} [Jy]'.format(source_dflux[i])
        print '                Peak flux = {0} [Jy/beam]'.format(source_peak[i])
        print '               No. pixels = {0}'.format(source_npix[i])
        print 'Flux weighted coordinates = ({0}, {1})'.format(world_coords[i][0], \
            world_coords[i][1])
        print '                      LAS = {0} [deg]'.format(source_LAS[i])
        print '              Source area = {0} [deg^2]'.format(source_area[i])
        if dist < 1.0:
            dist_print = dist * 60.0
            dist_unit = 'arcmin'
            if dist_print < 1.0:
                dist_print *= 60.0
                dist_unit = 'arcsec'
        else:
            dist_print = dist
            dist_unit = 'deg'
        print '        Offset from input = {0} [{1}]'.format(dist_print, \
            dist_unit)
        
    return source[i], source_flux[i], source_dflux[i], source_avg_flux[i], \
        source_area[i], source_npix[i], world_coords[i], bright_coords[i], \
        source_LAS[i], source_peak[i]



def force_measure_tree(fitsfile, RA, DEC, radius, rms, cutoff=3, LAS=True, \
                       verbose=False, radius_units='deg'):
    '''Measure flux within an aperture. 

    This can sum flux above a threshold or simply sum flux below a threshold
    to get a a limit.

    Parameters
    ----------
    fitsfile     : string or astropy.io.fits.HDUList
                   If string this should be the filename and path to a FITS file.
    RA           : float
                   Central RA in decimal degrees.
    DEC          : float
                   Central DEC in decimal degrees.
    radius       : float
                   Aperture within which to calculate flux in degree.
    rms          : float
                   Average RMS of the map.
    cutoff       : int, optional
                   Multiple of the RMS required for detection. Default is 3.
    LAS          : bool, optional
                   If True an LAS is calculated.
    verbose      : bool, optional
                   If True results are printed to the terminal.
    radius_units : {'deg', 'arcmin', 'arcsec'}, optional
                   Specifies the unit for `radius`.

    Returns
    -------
    int_flux     : float
                   The integrated flux density within the aperture of `radius` 
                   in Jansky.
    unc_flux     : float
                   The uncertainty in the integrated flux density.
    npix         : int
                   The number of pixels that were above `cutoff`*`rms`.
    area         : float
                   The area in degrees squared of the pixels above `cutoff`*`rms`.
    las          : float or string
                   If `LAS` is True, then the `las` is returned in degrees. If not,
                   the `las` returned is the string "NA".

    '''

    if isinstance(fitsfile, str):
        hdulist = fits.open(fitsfile)
        opened  = True
    elif isinstance(fitsfile, fits.HDUList):
        hdulist = fitsfile
        opened  = False
    else:
        raise TypeError('>>> Input file must be `pathtofile/file` or an' \
                        'astropy.io.fits.HDUList object.')

    farray, warray, beams_per_pixel, cd1, cd2, hdulist = \
        read_fits(hdulist=hdulist)
    naxis = hdulist[0].header['NAXIS']

    if radius_units == 'deg':
        radius_units = 1.0
    elif radius_units == 'arcmin':
        radius_units = 60.0
    elif radius_units == 'arcsec':
        radius_units = 3600.0
    else:
        raise TypeError('>>> `radius_units` must be one of [`deg`, '\
                        '`arcmin`, `arcsec`].')

    source_flux, source_xpixel, source_ypixel, source_coords = [], [], [], []

    for i in range(0, len(farray[:, 0])):
        for j in range(0, len(farray[0, :])):

            if farray[i, j] >= cutoff*rms:

                if naxis == 2:
                    coords = warray.all_pix2world(j, i, 0)
                elif naxis == 3:
                    coords = warray.all_pix2world(j, i, 1, 0)
                elif naxis == 4:
                    coords = warray.all_pix2world(j, i, 1, 1, 0)
                else:
                    raise ValueError('>>> NAXIS must be 2, 3, or 4.')

                diff = angular_distance((RA, DEC), (coords[0], coords[1]))

                if diff <= (radius/radius_units):

                    source_flux.append(farray[i, j])
                    source_xpixel.append(i)
                    source_ypixel.append(j)
                    source_coords.append(coords)

    int_flux = sum(source_flux) / beams_per_pixel
    unc_flux = rms * numpy.sqrt(len(source_flux) / beams_per_pixel)
    area     = len(source_flux) * abs(cd1*cd2)
    npix     = len(source_flux)

    if LAS:

        las = 0

        for i in range(0, len(source_coords)):
            for j in range(0, len(source_coords)):

                if angular_distance(source_coords[i], source_coords[j]) > las:

                    las = angular_distance(source_coords[i], source_coords[j])

    else:

        las = 'NA'

    if verbose:

        print('>>> WITCHWOOD has calculated the following parameters:')
        print('Integrated flux  = %f [Jy]' % int_flux)
        print('Error int. flux  = %f [Jy]' % unc_flux)
        print('Number of pixels = %i' % npix)
        print('Source area      = %f [deg^2]' % area)
        if LAS:
            print('Source LAS       = %f [deg]' % las)
        else:
            print('No LAS has been found.')

    return int_flux, unc_flux, npix, area, las






def region_flux(fitsfile, region, rms, r_index=0, sigma=2.6, out_ann=None, \
                out_ann_color="yellow", verbose=False):
    """Measure flux within region specified by DS9 region file.

    This does what `radioflux.py` does; written by M.~J. Hardcastle:
    http://www.extragalactic.info/~mjh/radio-flux.html

    This adapts a C++ implementation found here:
    http://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/


    Note that region vetices must be ordered.
    
    Parameters
    ----------
    fitsfile
    region  
    rms
    r_index
    sigma
    out_ann
    """

    def orientation(p1, p2, p3):
        """Get orientation of ordered triplet. 
        
        0 == collinear
        1 == clockwise
        2 == counter-clockwise
        """

        slop = (p2[1] - p1[1]) * (p3[0] - p2[0]) - \
               (p3[1] - p2[1]) * (p2[0] - p1[0])

        if slop < 0:
            orient = 2
        elif slop > 0:
            orient = 1
        else:
            orient = 0

        return orient


    def intersects(p1, q1, p2, q2):
        """Determine if line segment (p1, q1) intersects with (p2, q2)

        Uses orientation conditions.
        """

        def on_segment(p1, p2, p3):
            """Checks if p3 lies on segment (p1, p2)."""

            if ((p3[0] <= max(p1[0], p2[0])) and (p3[0] >= min(p1[0], p2[0])) and\
                (p3[2] <= max(p1[1], p2[1])) and (p3[1] >= min(p1[1], p2[1]))):

                return True

            else:

                return False


        inter = False


        # General case:
        if (orientation(p1, q1, p2) != orientation(p1, q1, q2)) and \
           (orientation(p2, q2, p1) != orientation(p2, q2, q1)):

           inter = True

        # Special cases:
        if (orientation(p1, q1, p2) == 0) and (on_segment(p1, q1, p2)):
            inter = True
        if (orientation(p1, q1, q2) == 0) and (on_segment(p1, q1, q2)):
            inter = True
        if (orientation(p2, q2, p1) == 0) and (on_segment(p2, q2, p1)):
            inter = True
        if (orientation(p2, q2, q1) == 0) and (on_segment(p2, q2, q1)):
            inter = True


        return inter



    def is_in(x, y, region_x, region_y, max_x):
        """Check if (x, y) is in region.

        (x, y) is considered in region if the followed is met:
        A line drawn horizontally to the right from (x, y) intersects 
        with an odd number of region edges. 
        """  
        
        coords_in_region = []

        for i in range(len(x)):

            p1, q1 = (x[i], y[i]), (x[i]+max_x, y[i])

            intersections = 0

            for j in range(len(region_x)):
                p2 = (region_x[j], region_y[j])
                if j == len(region_x)-1:
                    q2 = (region_x[0], region_y[0])
                else:
                    q2 = (region_x[j+1], region_y[j+1])

                if intersects(p1, q1, p2, q2):
                    intersections += 1

            if intersections%2 == 0:
                in_region = False
            else:
                in_region = True


            coords_in_region.append(in_region)

        return coords_in_region


    import pyregion

    if region.endswith(".reg"):
        # This is a ds9 region file, so will use pyregion for parsing.
        r = pyregion.open(region)[r_index].coord_list
        print r

        coords = []
        ra, dec = [], []
        for i in range(0, len(r), 2):
            print r[i]
            coords.append((r[i], r[i+1]))
            ra.append(r[i])
            dec.append(r[i+1])
            
    elif region.endswith(".ann"):
        raise ValueError("No support for Kvis annotation files currently.")

    else:
        # Assume region is a list of coordinate tuples.
        coords = region
        ra, dec = [], []
        for vertex in region:
            ra.append(vertex[0])
            dec.append(vertex[1])

    farray, warray, beams_per_pixel, cd1, cd2, hdulist = \
        read_fits(fitsfile)
    max_x = len(hdulist[0].data[:, 0]+1)
    
    if isinstance(rms, str): 
        rms_image = fits.getdata(rms)
        opened = True
    elif isinstance(rms, fits.HDUList): 
        rms_image =rms[0].data
        opened  = False
    else: 
        rms_image = None
        opened = False

    if opened:
        print("An rms image has been opened.")
        
        
    if rms_image is not None:
        # if (len(rms_image[:, 0]) != len(farray[:, 0])) or \
        #    (len(rms_image[0, :]) != len(farray[0, :])):
        if rms_image.shape != farray.shape:
            raise ValueError("Primary and rms images are different sizes.")


    pix_d, pix_r = warray.all_world2pix(ra, dec, 0)
    x_all, y_all, f_all, r_all = [], [], [], []
    for i in range(0, len(hdulist[0].data[:, 0])):
        for j in range(0, len(hdulist[0].data[0, :])):

            x_all.append(i)
            y_all.append(j)
            f_all.append(hdulist[0].data[i, j])
            if rms_image is not None:
                r_all.append(rms_image[i, j])
    


    cir = is_in(x_all, y_all, pix_r, pix_d, max_x)


    tree_fluxes = []
    tree_rmses  = []

    print tree_rmses

    final_ra, final_dec = [], []


    for i in range(len(cir)):
        if rms is not None:
            if rms_image is not None: r = r_all[i]
            else: r = rms
            if cir[i] and (f_all[i] >= sigma*r):
                print r
                tree_fluxes.append(f_all[i])
                tree_rmses.append(r)
                xx, yy = warray.all_pix2world(y_all[i], x_all[i], 0)
                final_ra.append(xx), final_dec.append(yy)
        else:
            if cir[i]:
                tree_fluxes.append(f_all[i])
                xx, yy = warray.all_pix2world(y_all[i], x_all[i], 0)
                final_ra.append(xx), final_dec.append(yy)

    print tree_rmses
    
    if out_ann is not None:
        if out_ann.endswith(".ann"):
            with open(out_ann, "w+") as g:
                g.write("COLOUR {0}\n".format(out_ann_color))
                for i in range(len(final_ra)):
                    g.write("DOT W {0} {1}\n".format(final_ra[i], final_y[i]))

    source_flux  = sum(tree_fluxes) / beams_per_pixel
    if rms is not None:
        source_dflux = (sum(tree_rmses) / beams_per_pixel) * \
                        numpy.sqrt(beams_per_pixel / float(len(tree_fluxes)))
    else:
        rms = numpy.nan

    if verbose:

        print(">>> WITCHWOOD has calculated the following parameters:")
        print("Integrated flux  = {0} [Jy]".format(source_flux))
        print("Error int. flux  = {0} [Jy]".format(source_dflux))
        print("Number of pixels = {0}".format(len(tree_fluxes)))


    return source_flux, source_dflux, tree_fluxes

