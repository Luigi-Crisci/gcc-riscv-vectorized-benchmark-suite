#pragma once

#include <iostream>
#include <fstream>

template<typename U>
struct type_t{
    using type = U;
};

template<typename T>
struct type_info{};

template<>
struct type_info<int32_t>{
    static constexpr auto name = "int32";
};

template<>
struct type_info<int64_t>{
    static constexpr auto name = "int64";
};
template<>
struct type_info<float>{
    static constexpr auto name = "float";
};
template<>
struct type_info<double>{
    static constexpr auto name = "double";
};


struct logfile{

    static constexpr auto header = "exp_type,type,size,precision,lmul,time\n";
    
    logfile() {
        std::cout << header;
        std::cout << std::fixed;
    }

    ~logfile(){
    }

    template<typename T> 
    void add_res(std::string exp_type, type_t<T> type_value, int size, int precision, int lmul, double time){
        std::cout << exp_type << ',' << type_info<T>::name << ',' << size << ',' << precision << ',' << lmul << ',' << time << "\n";
    }

};

logfile& get_logfile(){
    static logfile file;
    return file;
}