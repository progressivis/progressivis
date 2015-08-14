import urllib2

def wget_file(filename, url):
    attempts = 0
    while attempts < 3:
        try:
            response = urllib2.urlopen("http://example.com", timeout = 5)
            content = response.read()
            with open(filename, 'w' ) as f:
                f.write( content )
            break
        except urllib2.URLError as e:
            attempts += 1
            print type(e)
    return filename
