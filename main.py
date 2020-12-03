#!/usr/bin/env python
"""
Very simple HTTP server in python (Updated for Python 3.7)
Usage:
    ./dummy-web-server.py -h
    ./dummy-web-server.py -l localhost -p 8000
Send a GET request:
    curl http://localhost:8000
Send a HEAD request:
    curl -I http://localhost:8000
Send a POST request:
    curl -d "foo=bar&bin=baz" http://localhost:8000
"""
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from ruGPT import ruGPTModel

from cgi import parse_header, parse_multipart
from urllib.parse import parse_qs

model = ruGPTModel()

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

    def parse_POST(self):
        ctype, pdict = parse_header(self.headers['content-type'])
        if ctype == 'multipart/form-data':
            postvars = parse_multipart(self.rfile, pdict)
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = parse_qs(
                self.rfile.read(length).decode('cp1251'),
                keep_blank_values=1)
        else:
            postvars = {}
        return postvars

    def _html(self, message):
        """This just generates an HTML document that includes `message`
        in the body. Override, or re-write this do do more interesting stuff.
        """
        content = f"{message}"
        return content.encode("utf8")  # NOTE: must return a bytes object!

    def do_GET(self):
        self._set_headers()
        self.wfile.write(self._html("hi!"))

    def do_HEAD(self):
        self._set_headers()

    # В пост отправляется строка на обработку в ruGPT-3
    def do_POST(self):
        # Doesn't do anything with posted data
        self._set_headers()
        postvars = self.parse_POST()

        context_str = postvars['context'][0]
        context_b = context_str.encode('utf-8')
        context_str = context_b.decode('utf-8')

        length = int(postvars['length'][0])
        k = int(postvars['k'][0])
        p = float(postvars['p'][0])
        temperature = float(postvars['temperature'][0])
        rp = float(postvars['rp'][0])
        nrs = int(postvars['nrs'][0])
        seed = int(postvars['seed'][0])

        param ={"length": length,
                "temperature": temperature,
                "repetition_penalty": rp,
                "k": k,
                "p": p,
                "seed": seed,
                "num_return_sequences": nrs}
        model.set_hyper_params(params=param)
        output_str = model.inference(context=context_str)
        self.wfile.write(output_str.encode('utf-8'))


def run(server_class=HTTPServer, handler_class=S, addr="localhost", port=8000):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

    print(f"Starting httpd server on {addr}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run a simple HTTP server")
    parser.add_argument(
        "-l",
        "--listen",
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Specify the port on which the server listens",
    )
    args = parser.parse_args()
    run(addr=args.listen, port=args.port)