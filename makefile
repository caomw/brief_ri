CC=g++
CFLAGS= -std=c++11 -pthread `pkg-config --cflags opencv` -Wall -g
LIBS=`pkg-config --libs opencv`

TARGET=main
$(TARGET):$(TARGET).cpp brief_ri.o
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

brief_ri.o: brief_ri.cpp brief_ri.h
	$(CC) -c $< $(CFLAGS) $(LIBS) -o $@

clean:
	rm -rf *.o $(TARGET)
