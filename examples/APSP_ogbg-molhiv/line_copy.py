def copy_first_100_lines(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for _ in range(100):
            line = infile.readline()
            if not line:
                break
            outfile.write(line)


input_file = "data\shortest_paths.cor"
output_file = "data\shortest_paths_first_n.cor"

copy_first_100_lines(input_file, output_file)
