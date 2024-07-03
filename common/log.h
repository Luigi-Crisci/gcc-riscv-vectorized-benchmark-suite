#include <iostream>
#include <fstream>

struct logfile{

    static constexpr auto header = "exp_type, precision, lmul, time\n";
    
    logfile(std::string filename) {
        std::cout << header;
        std::cout << std::fixed;
    }

    ~logfile(){
    }


    void add_res(std::string exp_type, int precision, int lmul, double time){
        std::cout << exp_type << ',' << precision << ',' << lmul << ',' << time << "\n";
    }

};

logfile& get_logfile(std::string name = ""){
    static logfile file(name);
    return file;
}