import numpy as np
from astropy.io import fits

with fits.open("./xsect_phabs_angr.fits") as fxs:
    dxs = fxs[1].data
    energy = np.array(dxs["ENERGY"], dtype=np.float64)
    sigma = np.array(dxs["SIGMA"], dtype=np.float64)

np.savez_compressed("./xsect_phabs_angr.npz", energy=energy, sigma=sigma)


with fits.open("./xsect_phabs_aspl.fits") as fxs:
    dxs = fxs[1].data
    energy = np.array(dxs["ENERGY"], dtype=np.float64)
    sigma = np.array(dxs["SIGMA"], dtype=np.float64)

np.savez_compressed("./xsect_phabs_aspl.npz", energy=energy, sigma=sigma)


with fits.open("./xsect_tbabs_angr.fits") as fxs:
    dxs = fxs[1].data
    energy = np.array(dxs["ENERGY"], dtype=np.float64)
    sigma = np.array(dxs["SIGMA"], dtype=np.float64)

np.savez_compressed("./xsect_tbabs_angr.npz", energy=energy, sigma=sigma)


with fits.open("./xsect_tbabs_aspl.fits") as fxs:
    dxs = fxs[1].data
    energy = np.array(dxs["ENERGY"], dtype=np.float64)
    sigma = np.array(dxs["SIGMA"], dtype=np.float64)

np.savez_compressed("./xsect_tbabs_aspl.npz", energy=energy, sigma=sigma)


with fits.open("./xsect_tbabs_wilm.fits") as fxs:
    dxs = fxs[1].data
    energy = np.array(dxs["ENERGY"], dtype=np.float64)
    sigma = np.array(dxs["SIGMA"], dtype=np.float64)

np.savez_compressed("./xsect_tbabs_wilm.npz", energy=energy, sigma=sigma)


with fits.open("./xsect_wabs_angr.fits") as fxs:
    dxs = fxs[1].data
    energy = np.array(dxs["ENERGY"], dtype=np.float64)
    sigma = np.array(dxs["SIGMA"], dtype=np.float64)

np.savez_compressed("./xsect_wabs_angr.npz", energy=energy, sigma=sigma)
