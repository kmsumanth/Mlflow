import argparse




if __name__=='__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument("--name","-n",default='sumanth',type=str)
    arg.add_argument("--age","-a",default=25.0,type=float)

    parse_args=arg.parse_args()

    print(parse_args.name,parse_args.age)
    
