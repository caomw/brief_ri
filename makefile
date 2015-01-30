CC=g++
CFLAGS= -std=c++11 -pthread `pkg-config --cflags opencv` -Wall -g
LIBS=`pkg-config --libs opencv`

TARGET=main
$(TARGET):$(TARGET).cpp 
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

clean:
	rm -rf *.o $(TARGET)
