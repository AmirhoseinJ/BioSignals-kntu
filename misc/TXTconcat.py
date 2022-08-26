from pathlib import Path

open(r"C:\Users\****\all.txt", 'w').close()
for path in Path(r"C:\Users\****\ditt").rglob("A_*.txt"):
    txtfile = open(path, "rt")
    str = (txtfile.readline())
    txtfile.close()
    main = open(r"C:\Users\****\ditt\all.txt", 'a')
    to = str.join("test")
    main.write(to)
    main.close()
