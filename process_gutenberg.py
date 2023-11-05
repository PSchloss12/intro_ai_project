import sys
import os

start = "*** START OF THE PROJECT GUTENBERG"
end = "*** END OF THE PROJECT GUTENBERG"

def process(path):
    # print(f"data{os.sep}{path.split(os.sep)[-1].split('.')[0]}_parsed.txt")
    with open(f"data{os.sep}{path.split(os.sep)[-1].split('.')[0]}_parsed.txt","w", encoding='utf-8') as ofp:
        with open(path,"r", encoding='utf-8') as ifp:
            write = False
            for l in ifp:
                line = l.strip()
                if end in line:
                    write = False
                    break
                if write:
                    try:
                        ofp.write(line+"\n")
                    except UnicodeEncodeError as ex:
                        pass
                if start in line:
                    write = True

if __name__ == "__main__":
    for file in os.listdir("raw_data"):
        fname = f"raw_data\{file}"
        print(fname)
        process(fname)
