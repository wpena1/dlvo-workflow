out_xml = open('params.xml', 'w')
out_xml.write("<section name='OtherParameters' type='section' title='Other Parameters' expanded='false'>")
with open('inputs.yaml', 'r') as in_file:
    for line in in_file:
        tokens = line.split(':')
        var_name = tokens[0].strip()
        value = tokens[1].strip().strip("'")
        label = var_name.replace('_',' ')
        out_line = F"\t<param name='{var_name}' label='{label}' type='text' value='{value}' width='30%' help='if multiple  enter separated with space'></param>\n"
        print(out_line)
        out_xml.write(out_line)
out_xml.write("</section>")
out_xml.close()
