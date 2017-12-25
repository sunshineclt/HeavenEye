import xml.etree.ElementTree as ET


class XMLOperation:
    @staticmethod
    def read_from_file(filename):
        xmlp = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(filename, parser=xmlp)
        items = list(tree.getroot())[1]
        assert items.tag == "Items"
        data = {'nothing': 0}
        for item in items:
            imageName = item.attrib["imageName"]
            labels = []
            for label in item:
                labels.append((int(label.attrib["id"]),
                               float(label.attrib["score"]),
                               int(label.attrib["l"]),
                               int(label.attrib["t"]),
                               int(label.attrib["r"]),
                               int(label.attrib["b"])))
            data[imageName] = labels
        return data

    @staticmethod
    def write_to_file(data, mediaFile, filename):
        root = ET.Element('Message', attrib={"Version": "1.0"})
        ET.SubElement(root, "Info", attrib={"evaluateType": "4",
                                            "mediaFile": mediaFile})
        items = ET.SubElement(root, "Items")
        for index, key in enumerate(data):
            value = data[key]
            item = ET.SubElement(items, "Item", attrib={"imageName": key})
            for label in value:
                ET.SubElement(item, "Label", attrib={"id": str(label[0]),
                                                     "score": str(label[1]),
                                                     "l": str(label[2]),
                                                     "t": str(label[3]),
                                                     "r": str(label[4]),
                                                     "b": str(label[5])})
        tree = ET.ElementTree(root)
        tree.write(filename)


if __name__ == "__main__":
    XMLOperation.write_to_file({
        "1.jpg": [],
        "2.jpg": [(30, 0.8, 1, 2, 3, 4)],
        "3.jpg": [(30, 0.8, 5, 6, 7, 8), (30, 0.8, 9, 10, 11, 12)]
    }, "dongnanmeneast_15_1920x1080_30", "out.xml")
