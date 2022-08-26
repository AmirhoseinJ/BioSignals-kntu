import pathlib
from xml.dom import minidom

# with help from https://stackoverflow.com/questions/1912434/how-to-parse-xml-and-get-instances-of-a-particular-node-attribute
dom = minidom.parse('/home/****/Sources.xml')
elements = dom.getElementsByTagName('b:Title')
elements_tag = dom.getElementsByTagName('b:Tag')

print(f"There are {len(elements)} items:")

for element, element_tag in zip(elements, elements_tag):
    print(element_tag.firstChild.data)
    print(element.firstChild.data)
    print('\n')

print(elements[0].firstChild.data)


def path_list_generator():
    path_list = []
    for path in pathlib.Path(r'D:\project\edf').rglob('*.edf'):
        path_list.append(path)
    print("EDF Files in path: " + str(len(path_list)))
    return path_list


path_list_generator()
