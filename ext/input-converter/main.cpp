#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;

int main(int argc, char *argv[]) {   
    
    int MAX_CALLS = -1;

    if(argc < 2 || argc > 3) {
        printf("Error! Wrong arguments specified.\nUsage: %s <input_file> <max_calls>(optional)\n", argv[0]);
        exit(1);
    }
    
    if(argc == 3) {
    	MAX_CALLS = atoi(argv[2]);
    }

    char * infile = argv[1];
    ifstream file(infile);
    
    if(file.is_open()) {
        
        int num_calls = 0;
        
        string line;
        while(getline(file, line)) {
            
            cout << line << endl;
            
            unsigned long a_x;
            unsigned long a_y;
            
            unsigned int f1;
            int f2;
            int f3;
            int f4;
            
            while (getline(file, line)) {                
                if (line == "EOR" ) {
                    cout << "EOR" << endl; // EOR line
                    break;
                }
                
                stringstream line_stream;
                line_stream << line;
                
                line_stream >> f1;
                line_stream >> f2;
                line_stream >> f3;
                line_stream >> f4;
                
                a_x = ((unsigned long)f1 << 32) | f2;
				a_y = ((unsigned long)f3 << 32) | f4;
                
                cout << a_x << "\t" << a_y << endl;
            }
            num_calls++;
            
            if (num_calls == MAX_CALLS) {
                break;
            }
        }
        
    } else {
        cout << "Error opening input file!" << endl;
    }

}
