# Clone the repository
```bash
git clone https://github.com/NikolaPeric286/JPEG-encoder
cd JPEG-encoder
```
# Initialize submodules
```bash
git submodule update --init 
```
# Configure and build
```bash
mkdir build
cd build
cmake ..
make
```

#testing
```bash
cd ..
unzip data.zip
cd output
./jpeg-encoder-cli ../data/*.tga
gimp ../data/*.jpg
```
