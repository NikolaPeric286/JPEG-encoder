# JPEG-encoder
JPEG-Encoder is a lightweight C++ image compression utility built from scratch to encode raw pixel data into the JPEG format. Designed as both a learning project and a practical tool, it implements the full JPEG pipeline — including color-space conversion, chroma subsampling, DCT transformation, quantization, and Huffman coding — without relying on external image libraries. The command-line interface supports batch processing of uncompressed image formats such as .TGA and,  This project focuses on clarity, modularity, and cross-platform compatibility, making it ideal for studying how JPEG encoding works at a low level or integrating a minimal encoder into larger systems.     

### Current compatible file types:

 - .tga

## How to build
## **Linux**  


```bash
git clone https://github.com/NikolaPeric286/JPEG-encoder
cd JPEG-encoder
git submodule update --init 
mkdir build
cd build
cmake ..
make
```

#### Unit Tests  

```bash

cd ../output
./unit_tests

```

#### Testing With Files
```bash
cd ..
unzip data.zip
cd output
./jpeg-encoder-cli ../data/*.tga
gimp ../data/*.jpg
```
