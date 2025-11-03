# Clone the repository
git clone https://github.com/NikolaPeric286/JPEG-encoder
cd JPEG-encoder

# Initialize submodules
git submodule update --init 

# Configure and build
mkdir build
cd build
cmake ..
make
