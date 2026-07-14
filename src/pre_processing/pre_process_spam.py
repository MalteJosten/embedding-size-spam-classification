import os
import json
import uuid
import re
import argparse
from email import message_from_string
from email.header import decode_header


def prepare_destination_directories(des_dir, year):
    if os.path.exists(des_dir):
        for file in os.listdir(des_dir):
            if file.startswith(year):
                os.remove(os.path.join(des_dir, file))
    else:
        os.makedirs(des_dir)

    if os.path.exists(f"{des_dir}_non_utf8"):
        for file in os.listdir(des_dir):
            if file.startswith(year):
                os.remove(os.path.join(des_dir, file))
    else:
        os.makedirs(f"{des_dir}_non_utf8")


def preprocess_mails(src_dir, des_dir, year):
    file_counter = 0
    non_utf8_counter = 0
    mappings = {}

    for sub_dir, dirs, files in os.walk(src_dir):
        for month_dir in dirs:
            for file in os.listdir(os.path.join(sub_dir, month_dir)):
                try:
                    with open(os.path.join(sub_dir, month_dir, file), "r", encoding="utf-8") as raw_file:
                        content = raw_file.read()
                        is_non_utf8 = False

                        if content == None or len(content) == 0:
                            continue

                        mail = message_from_string(content)

                        split_content = content.split("\n\n")
                        header = split_content[0]
                        body = "\n\n".join(split_content[1:])

                        if header == None or len(header) == 0:
                            continue

                        if body == None or len(body) == 0:
                            continue


                        new_file = ""
                        new_file += "To: to@example.com\n"
                        new_file += "From: from@example.com\n"

                        for line in header.split("\n"):
                            if line.lower().startswith("subject"):
                                subject, charset = decode_header(mail["Subject"])[0]
                                if charset and charset.lower() != "utf-8":
                                    is_non_utf8 = True
                                    if isinstance(subject, bytes):
                                        subject = subject.decode(charset, errors="replace")

                                new_file += f"{line}\n"

                        new_file += "\n\n"

                        re_encoded_parts = []
                        for part in mail.walk():
                            if part.is_multipart():
                                continue

                            content_type = part.get_content_type()
                            if not content_type.startswith("text/"):
                                continue

                            charset = part.get_content_charset() or "utf-8"

                            payload = part.get_payload(decode=True)
                            if payload is None:
                                continue

                            try:
                                text = payload.decode(charset, errors="replace")
                            except LookupError:
                                is_non_utf8 = True
                                text = payload.decode("utf-8", errors="replace")

                            re_encoded_parts.append(text)
                        
                        body = "\n".join(re_encoded_parts)

                        # preserve links
                        body = re.sub(r'<a [^>]*href=["\'](https?://[^"\']+)["\'][^>]*>(.*?)</a>', r'\2 (\1)', body, flags=re.IGNORECASE | re.DOTALL)
                        # remove html
                        body = re.sub(r'<[^>]+>', '', body)
                        # remove newlines
                        body = re.sub(r'\n', '', body)
                        # remove multiple spaces
                        body = re.sub(r'[ ]{2,}', ' ', body)

                        new_file += body

                        if body == None or len(body) == 0:
                            continue

                        if new_file == "" or new_file == None:
                            continue

                except:
                    continue

                try:
                    new_file_name = f"{year}_{uuid.uuid4()}.eml"

                    if is_non_utf8:
                        new_file_path = os.path.join(f"{des_dir}_non_utf8", new_file_name)
                    else:
                        new_file_path = os.path.join(des_dir, new_file_name)

                    with open(new_file_path, "w") as w_file:
                        w_file.write(new_file)

                        mappings[file] = new_file_name

                except:
                    continue

                file_counter += 1
                if is_non_utf8:
                    non_utf8_counter += 1
    return file_counter, non_utf8_counter, mappings


def write_mappings(mappings_file, mappings):
    with open(mappings_file, "w") as m_file:
        m_file.write(json.dumps(mappings, indent=2))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--src", type=str, required=True)
    argparser.add_argument("--des", type=str, required=True)
    argparser.add_argument("--years", nargs="+", type=int, required=True)

    args = argparser.parse_args()

    years = range(args.years[0], args.years[1])
    src_dir = args.src
    des_dir = args.des
    mapping_file = "file_mappings.json"

    for year in years:
        src_dir_y = f"{src_dir}{year}"

        prepare_destination_directories(des_dir, str(year))
        file_counter, non_utf, mappings = preprocess_mails(src_dir_y, des_dir, str(year))
        write_mappings(mapping_file, mappings)
        print(f"{src_dir_y} > {file_counter} ({non_utf})")


if __name__ == "__main__":
    main()
