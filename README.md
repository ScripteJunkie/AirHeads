# AirHeads
## The Lovecraftian Horror of Air Hockey, Energy Drinks, and Brilliant Minds

![Example tracking](https://github.com/ScripteJunkie/AirHeads/blob/main/server/public/tracked.jpg)

### Install:
> Clone and open AirHeads library.
```
git clone --recursive https://github.com/rirmps/AirHeads.git
cd AirHeads
```

### Setup:
> Setup virtal environment.
```
conda create -n airhead python=3.8
conda activate airhead
python ./lib/depthai/install_requirments.py
```

### Alignment:
> Center table with displayed target.
```
python ./src/align.py
```

### Calibration:
> Adjust sliders to determine colors and click table borders to find edge bounds.
> Hit 's' when finished and change trackOUT file values based on console output.
```
python ./src/colorfinder.py
```

### Run:
> Displays both mask and tracked frames.
```
python ./src/trackOUT.py
```

