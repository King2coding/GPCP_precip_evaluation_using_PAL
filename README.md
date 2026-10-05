# Multi-reference evaluation of ocean precipitation products

Research code for evaluating satellite, reanalysis, and merged precipitation
products over the oceans using multiple independent reference datasets. The
workflows integrate PAL observations, moored buoys, atolls, and OceanRAIN and
support comparisons involving GPCP, IMERG, ERA5, and MERRA-2 across daily,
monthly, climatological, regional, and intensity-based perspectives.

## Repository status

This is a code-only archive for an active manuscript project. The scientific
scripts and their development history are preserved without refactoring.
Observational data, derived matchups, figures, manuscript files, and other
generated outputs are intentionally excluded.

## Workflow coverage

- preparation and availability assessment of PAL, buoy, atoll, and OceanRAIN data;
- matching in-situ observations with gridded precipitation products;
- categorical and quantitative evaluation statistics;
- intensity-stratified, spatial, seasonal, and anomaly analyses;
- western-Pacific and other regional diagnostics; and
- production of manuscript-oriented figures and summary tables.

The repository retains several generations of the workflow. Files with labels
such as `prime`, `quick`, `test`, or regional descriptors record specific
analysis stages and should not be treated as interchangeable entry points.

## Associated manuscript

Kumah, K. K., Behrangi, A., Huffman, G. J., Adler, R. F., Gu, G., Song, Y.,
Zandi, O., Bolvin, D. T., Nelkin, E. J., Funk, C. C., and Peterson, P. (2026).
*A Multi-Reference Assessment of Ocean Precipitation Products by Integrating
PAL, Buoys, Atolls, and OceanRAIN.* Manuscript submitted.

This citation and status match the author's public research website as of
October 2026. Until formal publication, the manuscript status should always be
stated explicitly.

## Data and execution

The scripts expect externally obtained research datasets and, in several cases,
paths from the original rain-server environment. Those paths remain unchanged
to preserve the analysis record. See [`DATA.md`](DATA.md) before attempting to
run a workflow.

Common dependencies include `numpy`, `pandas`, `xarray`, `scipy`, `matplotlib`,
`seaborn`, and geospatial or NetCDF libraries. Exact versions vary among
workflow generations and were not consistently recorded.

## License

No software license has yet been assigned. Contact the author before reuse or
redistribution.

## Contact

Kwabena Kingsley Kumah — [GitHub profile](https://github.com/King2coding)
