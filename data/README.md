# Sample events

Three events and their climatologies, each the whole northern hemisphere on the
nine pressure levels the inversion works on.

| file | case | centre | time |
|---|---|---|---|
| `era5_blocking.nc` | ERA5, the California blocking ridge | 43.5 N, 226.5 E | 2025-01-08 00 UTC |
| `cesm_midlatitude.nc` | CESM2-LENS2 blocking peak | 42.4 N, 37.8 E | 1985-02-25 18 UTC |
| `cesm_greenland.nc` | CESM2-LENS2 blocking peak over northern Greenland | 78.6 N, 295.7 E | 2002-01-06 12 UTC |

Each has a `*_clim.nc` beside it: the climatology for the same day and hour, which
is what the event is an anomaly against. `events.json` holds the centres in a form
the examples read.

The Greenland case is the one a limited-area method cannot centre a box on. Half of
the surroundings of a 79 N event lie across the pole, where a latitude-longitude
box has nothing to say.

## Conventions

Every file uses the same ones, and the reader assumes them:

    lat      ascending, 0 to 90 N
    lon      ascending, starting at 0
    level    descending in pressure, 1000 hPa first
    z        geopotential height [m] -- not geopotential
    t        temperature [K] -- not potential temperature
    u, v     wind components [m s-1]

Two of these are worth stating twice. The reanalysis archives geopotential and the
model archives height; the division by g is done once, here, so nothing downstream
has to remember which it received. And the inversion forms potential temperature
itself: handing it potential temperature instead of temperature gives a state that
is wrong by one factor of the Exner function and still looks entirely plausible.

Provenance -- the archive each sample was cut from, and which step or climatology
slot -- is in the `source` attribute of every file.
