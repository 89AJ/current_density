# Current density calculations with gpaw
WORK IN PROGRESS!!

Current density is calculated using hydrogen electrodes. 

## Clone the repository
```
git clone https://github.com/89AJ/current_density.git
```

## Test that it works!
```
cd current_density/src/
```
```
gpaw-basis H -t sz
```
```
python dump_ham.py
```
```
python calc_local.py
```
```
jmol c8/current.spt
```


## Run your own calculations
goto data folder and create a new folders to your molecule
```
./make_dirs.sh <folder name>
```
Move your xyz file in there.
```
mv path/to/your/<name>.xyz path/to/current_density/data/<folder name>/ 
```
go to src
```
cd current_density/src/
```
dump hamilitonian and basis
```
python dump_ham.py --path path/to/current_density/data/<folder name>/ --basis sz (dzp/tzdp/..)
```
Run current density calculation
```
python calc_local.py  --path path/to/current_density/data/<folder name>/
```
view using jmol
```
jmol c8/current.spt
```

## Requirements:
On Mac (& Linux?):

Install conda from website
```
pip install ase
```
```
pip install gpaw
```

On Windows:

Good luck!





