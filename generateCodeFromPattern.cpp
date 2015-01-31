#include <iostream>
#include <fstream>


using namespace std;

int main(){
	ifstream ifs( "pattern", ifstream::in );
	ofstream ofs( "pattern_", ofstream::out );
	int x1, y1, x2, y2;
	ofs << "{" << endl;
	for ( size_t i = 0; i < 512; ++i ){
		ifs >> x1 >> y1 >> x2 >> y2;
		if ( i != 511 ){
			ofs << x1 << "," << y1 << "," << x2 << "," << y2 << ",";
		}else{
			ofs << x1 << "," << y1 << "," << x2 << "," << y2 << "}";
		}
	}
	ifs.close();
	ofs.close();
}
