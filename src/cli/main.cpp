// main.cpp
//
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <thread>
#include "../backend/jpgenc.hpp"

#ifndef MAX_THREAD_COUNT
#define MAX_THREAD_COUNT 8
#endif

int main(int argc, char** argv){

    std::queue<std::string> file_queue;
    std::string output_file_name;
    bool set_output = false;
    //std::string file_name = "/home/nikolaperic/vscodeProjectFolders/JPEG-encoder/data/TGA_examples/EXAMPLE_part1.tga";
    
    for(int i = 1; i < argc; i++){

        
        if(argv[i][0] != '-' ){
            file_queue.push(argv[i]);
        }
        else{
            if(argv[i][1] == 'o'){
                output_file_name = argv[i+1];
                i++;
                set_output = true;
            }
        }
    }
    
    if(file_queue.size() > 1 && set_output){
        std::cerr << "Too many arguments\n";
        return 1;
    }

    std::queue<std::thread> threads;
    
    while (!file_queue.empty()){


        std::string file_name = file_queue.front();
        file_queue.pop();

        RGBImage input_rgb = getImageContentTGA(file_name.c_str());

        file_name.erase(file_name.length() - 4); // removes .tga 
        file_name.append(".jpg");
        output_file_name = (set_output)? output_file_name:file_name;


        std::cout << "Encoding " << file_name << std::endl;
        threads.emplace([img = std::move(input_rgb), out = std::move(output_file_name)]() mutable{
            std::ofstream out_file(out, std::ios::trunc | std::ios::binary);
            if(!out_file){
                std::cerr << "ERROR failed to open output file " << out << "\n";
            }

            encodeRGBImage(img,out_file);
            out_file.close();
        });

        if(threads.size() >= MAX_THREAD_COUNT){
            threads.front().join();
            threads.pop();
        }

        
    }
    

    while(!threads.empty()){
        threads.front().join();
        threads.pop();
    }
   
    
}