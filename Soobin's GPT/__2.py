import base64
from io import BytesIO
import time
import json
import os
from PIL import Image, ImageDraw, ImageFont
import pymysql
import urllib3
import boto3
import subprocess
import requests
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    concatenate_videoclips,
    AudioFileClip,
    TextClip,
    CompositeVideoClip,
)
import PIL

PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
from moviepy.config import change_settings
import moviepy
import tempfile

font_path = "/opt/layer/usr/share/fonts/custom/kbfont.ttf"

os.environ["TMPDIR"] = tempfile.gettempdir()
os.environ["IMAGEMAGICK_BINARY"] = "/opt/layer/bin/convert"
os.environ["MAGICK_CONFIGURE_PATH"] = "/opt/layer/etc/ImageMagick-6"
os.environ["LD_LIBRARY_PATH"] = (
    "/opt/layer/lib:/opt/layer/usr/lib:/lib64:/usr/lib64:/usr/local/lib"
)
# os.environ["LD_LIBRARY_PATH"] = "/opt/layer/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
# os.environ["MAGICK_CONFIGURE_PATH"] = "/opt/layer/lib/ImageMagick-6"

# # ImageMagick 실행 파일 경로 설정
change_settings({"IMAGEMAGICK_BINARY": "/opt/layer/bin/convert"})

# OpenAI API 설정
OPENAI_API_URL_IMAGE = "https://api.openai.com/v1/images/generations"
OPENAI_API_URL_STT = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_API_URL_TTS = "https://api.openai.com/v1/audio/speech"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# client 설정
S3_BUCKET = os.environ.get("bucket")
s3_client = boto3.client("s3")
polly = boto3.client("polly")


# MySQL 데이터베이스 연결
def get_connection():
    db_host = "pea-hrd-dev-az1-db-01.cpqwgpfv5pha.ap-northeast-2.rds.amazonaws.com"
    db_user = "admin"
    db_password = "ehrd*37620"
    db_database = "PEA_MY"

    return pymysql.connect(
        host=db_host, user=db_user, password=db_password, database=db_database
    )


def find_ctt_title_in_db(cttmngno):
    conn = get_connection()
    cursor = conn.cursor()

    # 컨텐츠 관리번호로 데이터베이스 조회
    query = (
        "SELECT CTT_TITLE FROM PEA_MY.CMS_CTT_MST WHERE CTT_MNG_NO = %s AND STT='00'"
    )
    cursor.execute(query, (cttmngno,))
    result = cursor.fetchone()

    # 연결 종료
    cursor.close()
    conn.close()

    return result[0]


def find_summary_in_db(cttmngno):
    conn = get_connection()
    cursor = conn.cursor()

    # 컨텐츠 관리번호로 데이터베이스 조회
    query = "SELECT CTT_SUMM_SCRIPT FROM PEA_MY.CMS_CTT_TRANS WHERE CTT_MNG_NO = %s AND STT='00'"
    cursor.execute(query, (cttmngno,))
    result = cursor.fetchone()

    # 연결 종료
    cursor.close()
    conn.close()

    return result[0]


def find_keyword_id_in_db(cttmngno):
    conn = get_connection()
    cursor = conn.cursor()

    # 컨텐츠 관리번호로 데이터베이스 조회
    query = "SELECT GROUP_CONCAT(CTT_KEYWORD) FROM PEA_MY.CMS_CTT_KEYWORD WHERE CTT_MNG_NO = %s AND STT='00'"
    cursor.execute(query, (cttmngno,))
    result = cursor.fetchone()

    # 연결 종료
    cursor.close()
    conn.close()

    return result


def find_LLMResult_id_in_db(cttmngno):
    conn = get_connection()
    cursor = conn.cursor()

    # 컨텐츠 관리번호로 데이터베이스 조회
    query = (
        "SELECT LLM_RESULT FROM PEA_MY.CMS_CTT_TRANS WHERE CTT_MNG_NO = %s AND STT='00'"
    )
    cursor.execute(query, (cttmngno,))
    result = cursor.fetchone()
    # 연결 종료
    cursor.close()
    conn.close()

    return result[0]


def find_cttpath_in_db(cttmngno):
    conn = get_connection()
    cursor = conn.cursor()

    # 컨텐츠 관리번호로 데이터베이스 조회
    query = (
        "SELECT CTT_PATH FROM PEA_MY.CMS_CTT_TRANS WHERE CTT_MNG_NO = %s AND STT='00'"
    )
    cursor.execute(query, (cttmngno,))
    result = cursor.fetchone()
    # 연결 종료
    cursor.close()
    conn.close()

    return result[0]


def update_ctt_path_in_db(ctt_mng_no, uploadKey):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            sql_update = """
                UPDATE PEA_MY.CMS_CTT_TRANS
                SET PATH_GEN_VIDEO_B = %s
                WHERE CTT_MNG_NO = %s and STT = '00'
            """
            cursor.execute(sql_update, (uploadKey, ctt_mng_no))
            print("[SUCCESS] update mysql: upload path", ctt_mng_no)
            conn.commit()

    finally:
        conn.close()


# Polly로 MP3 생성
def generate_audio(script, S3_BUCKET, output_key):
    response = polly.synthesize_speech(
        Text=script, OutputFormat="mp3", VoiceId="Seoyeon"  # Polly에서 사용할 음성 선택
    )
    audio_file_path = "/tmp/audio.mp3"
    with open(audio_file_path, "wb") as f:
        f.write(response["AudioStream"].read())

    # S3에 업로드
    s3_client.upload_file(audio_file_path, S3_BUCKET, output_key)
    return f"{output_key}"


def generate_and_upload_tts(text, S3_BUCKET, audio_file_s3_key):
    """
    OpenAI TTS API를 사용하여 음성을 생성하고, AWS S3에 업로드하는 함수
    :param text: 변환할 텍스트
    :param s3_filename: S3에 저장될 파일 이름 (예: "output.mp3")
    :return: S3에 저장된 MP3 URL
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    data = {"model": "tts-1", "voice": "echo", "input": text}

    # OpenAI API 호출
    http = urllib3.PoolManager()
    response = http.request(
        "POST", OPENAI_API_URL_TTS, headers=headers, body=json.dumps(data)
    )

    if response.status == 200:
        # MP3 파일 저장
        local_mp3_path = "/tmp/audio.mp3"
        with open(local_mp3_path, "wb") as audio_file:
            audio_file.write(response.data)

        # S3에 업로드
        s3_client = boto3.client("s3")
        s3_client.upload_file(local_mp3_path, S3_BUCKET, audio_file_s3_key)

        return audio_file_s3_key
    else:
        print(f"Error: {response.status} - {response.data}")
        return None


# DALL-E 이미지 생성 함수
def genDalleImage(keywords, num_images):
    print("keyword:::", keywords)

    img_urls = []

    for keyword in keywords:

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
        data = {
            "model": "dall-e-3",  # dall-e-2
            # "prompt": f"A photorealistic representation of the {keyword}",
            "prompt": f"{keyword}",
            "n": num_images,
            "size": "1024x1792",
            # "size": "512x512"
            "style": "vivid",
        }

        http = urllib3.PoolManager()
        response = http.request(
            "POST", OPENAI_API_URL_IMAGE, headers=headers, body=json.dumps(data)
        )

        if response.status == 200:
            result = json.loads(response.data.decode("utf-8"))
            urls = [entry["url"] for entry in result.get("data", []) if "url" in entry]
            img_urls.append(", ".join(urls))
        else:
            print(f"Error: {response.status} - {response.data}")
    return img_urls


# DALL-E 이미지 생성 함수
def genDalleImageWithPrompts(scenes, num_images):
    print("scenes:::", scenes)

    img_urls = []

    for scene in scenes:

        prompt = scene["image_prompt"]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
        data = {
            "model": "dall-e-3",  # dall-e-2
            "prompt": prompt,
            "n": num_images,
            "size": "1024x1792",
            # "size": "512x512"
            "style": "vivid",
        }

        http = urllib3.PoolManager()
        response = http.request(
            "POST", OPENAI_API_URL_IMAGE, headers=headers, body=json.dumps(data)
        )

        if response.status == 200:
            result = json.loads(response.data.decode("utf-8"))
            urls = [entry["url"] for entry in result.get("data", []) if "url" in entry]
            img_urls.append(", ".join(urls))
        else:
            print(f"Error: {response.status} - {response.data}")
    return img_urls


# DALL-E 이미지  다운로드 함수
def download_image_from_url(image_url, output_path):
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.save(output_path, format="PNG")
            return output_path
        else:
            print(f"Failed to download image from {image_url}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def download_and_upload_to_s3(image_urls, cttmngno):
    s3_paths = []

    for idx, image_url in enumerate(image_urls):
        s3_key = (
            f"temp_LXP/genShortForms/output/image/{cttmngno}_{int(time.time())}.png"
        )
        try:
            # 이미지 다운로드
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            # S3 업로드
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=response.content,
                ContentType="image/png",
            )

            # 업로드된 S3 경로 저장
            s3_paths.append(f"s3://{S3_BUCKET}/{s3_key}")
            print(f"Image uploaded to S3: {s3_paths[-1]}")

        except requests.RequestException as e:
            print(f"Failed to download image from {image_url}: {e}")
        except Exception as e:
            print(f"Failed to upload image to S3: {e}")

    return s3_paths


# 비디오 생성 함수
def makeShortformWithImages(
    image_urls, ctt_title, cttmngno, audio_file_s3_key, output_file, vtt_json
):
    try:
        # DALL-E 이미지 다운로드
        tmp_image_paths = []
        for idx, image_url in enumerate(image_urls):
            tmp_image_path = f"/tmp/tmp_image_{idx + 1}.png"
            if download_image_from_url(image_url, tmp_image_path):
                tmp_image_paths.append(tmp_image_path)
        print("tmp_image_paths", tmp_image_paths)

        # 오디오 클립 로드
        audio_file = "/tmp/audio.mp3"
        audio = AudioFileClip(audio_file)
        audio_duration = audio.duration
        print("[ing] AI로 생성한 스크립트 요약 본으로 음성 파일(mp3) 생성")

        # 이미지 당 표시 시간 계산
        image_duration = audio_duration / len(image_urls)

        # 이미지 클립 생성
        image_clips = []
        for img in tmp_image_paths:

            # 텍스트 이미지 생성 (별도 파일로 저장)
            text_image_path = f"/tmp/text_overlay_{idx}.png"
            create_text_image(
                ctt_title,
                text_image_path,
                width=580,
                font_size=32,
                padding=20,
                radius=20,
            )

            # 원본 이미지 클립
            clip = (
                ImageClip(img)
                .set_duration(image_duration)
                .set_fps(24)
                .resize((720, 1280))
            )
            clip = clip.resize(
                lambda t: 1 + 0.4 * t / image_duration
            )  # 애니메이션 효과 추가 (줌 인/아웃, 더 빠르게 조정) 점점 확대 속도 증가

            # 텍스트 이미지 클립
            text_clip = ImageClip(text_image_path, ismask=False).set_duration(
                image_duration
            )
            text_clip = text_clip.set_position(
                ("center", 1280 * 0.2)
            )  # 텍스트를 상단에 배치

            # 이미지와 텍스트 결합
            composite = CompositeVideoClip([clip, text_clip])
            image_clips.append(composite)

        # 이미지 클립 이어 붙이기
        final_video = concatenate_videoclips(image_clips, method="compose")
        final_video = final_video.set_audio(audio)

        if vtt_json:
            timestamps, captions = vtt_json

            # 자막 클립 생성
            subtitle_clips = []
            for idx, ((start, end), caption) in enumerate(zip(timestamps, captions)):
                subtitle_image_path = f"/tmp/subtitle_{idx}.png"
                create_text_image(
                    caption,
                    subtitle_image_path,
                    width=720,
                    font_size=30,
                    padding=20,
                    radius=0,
                )
                print(f"Subtitle image saved: {subtitle_image_path}")

                # 자막 이미지 클립 생성
                subtitle_clip = ImageClip(
                    subtitle_image_path, ismask=False
                ).set_duration(end - start)
                subtitle_clip = subtitle_clip.set_position(
                    ("center", final_video.h * 0.78)
                )  # 하단 20% 위치
                subtitle_clip = subtitle_clip.set_start(start).set_end(end)

                subtitle_clips.append(subtitle_clip)
            print("[ing] 이미지 내 자막 배치")

            # 자막과 동영상 합성
            final_video = CompositeVideoClip([final_video, *subtitle_clips])
            print("[ing] 자막과 동영상 합성")

        # 결과 비디오 저장
        final_video.write_videofile(
            output_file,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="/tmp/temp_audiofile.m4a",  # 임시 오디오 파일
            remove_temp=True,  # 임시 파일 제거
        )
        print(f"비디오가 성공적으로 생성되었습니다: {output_file}")
        return output_file

    except Exception as e:
        print(f"비디오 생성에 오류가 발생했습니다: {e}")
        return None

    finally:
        # 임시 파일 정리
        temp_files = [f"/tmp/tmp_text_{idx + 1}.txt" for idx in range(len(image_urls))]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"임시 파일 {temp_file} 삭제 완료")


def generate_summary_vtt(audio_file_s3_key):
    audio_file = "/tmp/audio.mp3"

    try:
        s3_client.download_file(os.environ.get("bucket"), audio_file_s3_key, audio_file)
        print("Audio file downloaded successfully.")

    except Exception as e:
        print(f"Audio file 오류가 발생했습니다: {e}")
        return None

    file_path = audio_file

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    # Request payload
    data = {
        "model": "whisper-1",
        "timestamp_granularities": ["word"],
        "response_format": "verbose_json",
        "temperature": 0.2,
        "prompt": "한국어 맞춤법 틀리지 않게 해줘",
        "language": "ko",
    }

    # File upload
    with open(file_path, "rb") as audio_file:
        files = {"file": audio_file}

        # Send POST request
        response = requests.post(
            OPENAI_API_URL_STT, headers=headers, data=data, files=files
        )

    # Check response
    if response.status_code == 200:
        segments = response.json()["segments"]

        if segments:
            timestamps = []
            captions = []

            for segment in segments:
                start = segment.get("start")
                end = segment.get("end")
                text = segment.get("text")
                text = split_long_lines(text, 25)

                # Append timestamp and caption
                timestamps.append((start, end))
                captions.append(text)

            return timestamps, captions
        else:
            print(f"There is no segments")
    else:
        print(f"Error: {response.status_code}, {response.text}")


def create_text_image(text, output_path, width, font_size, padding, radius):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    # 텍스트 크기 측정
    dummy_img = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 가로(width)는 고정, 세로(height)는 자동 조절 (텍스트 높이 + 패딩)
    height = text_height + padding * 2

    # 둥근 모서리 마스크 생성
    if radius > 0:
        mask = Image.new("L", (width, height), 0)
        mask_draw = ImageDraw.Draw(mask)

        # 둥근 모서리 적용
        mask_draw.rectangle((radius, 0, width - radius, height), fill=255)  # 중앙 부분
        mask_draw.rectangle((0, radius, width, height - radius), fill=255)  # 상하 부분
        mask_draw.ellipse((0, 0, radius * 2, radius * 2), fill=255)  # 좌상단
        mask_draw.ellipse(
            (width - radius * 2, 0, width, radius * 2), fill=255
        )  # 우상단
        mask_draw.ellipse(
            (0, height - radius * 2, radius * 2, height), fill=255
        )  # 좌하단
        mask_draw.ellipse(
            (width - radius * 2, height - radius * 2, width, height), fill=255
        )  # 우하단

        # 배경 이미지 생성 (반투명 검은색 배경)
        img = Image.new("RGBA", (width, height), color=(0, 0, 0, 100))
        # 마스크 적용
        img.putalpha(mask)
        draw = ImageDraw.Draw(img)

    else:
        img = Image.new("RGBA", (width, height), color=(0, 0, 0, 200))
        draw = ImageDraw.Draw(img)

    # 텍스트 중앙 정렬
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2 - 3

    draw.text((text_x, text_y), text, font=font, fill="white")

    # 이미지 저장
    img.save(output_path)


def split_long_lines(text, max_length):
    words = text.split()  # 텍스트를 단어 단위로 나눔
    lines = []
    current_line = ""

    for word in words:
        # 현재 줄에 단어를 추가해도 길이가 max_length를 초과하지 않으면 추가
        if len(current_line) + len(word) + 1 <= max_length:
            if current_line:
                current_line += " "  # 단어 사이에 띄어쓰기 추가
            current_line += word
        else:
            # 초과하면 현재 줄을 저장하고 새 줄 시작
            lines.append(current_line)
            current_line = word

    # 마지막 줄 추가
    if current_line:
        lines.append(current_line)

    return "\n".join(lines)


# # # Lambda 핸들러 함수
def lambda_handler(event, context):
    print("luckykyy")
    print(event)

    # cttmngno = 'c7e1acf292f14cd8ae7e'
    # cttmngno = '84bac654e33c42bfb08f'
    # cttmngno = '32a3b6c60f8e4163879a'
    # 20250220 kgs
    # cttmngno = 'Y00327807'
    cttmngno = event["cttMngNo"]
    # cttmngno = event['mngSn']
    print("cttmngno:" + cttmngno)

    # # lambda연동 테스트시작
    # print ("job test !! ");
    # #작업 완료시 SNS 전송
    # sns = boto3.client("sns")
    # resp_sns = sns.publish(
    #     TopicArn=os.environ.get("SNS_ARN_JOBFINISH"),
    #     Message=json.dumps(event),
    #     Subject='task Completion GEN AI shorts'
    # )
    # print ("SNS publish resp", resp_sns)
    # return {"statusCode": 200, "body": "receive SNS call test !!!"}
    # # lambda연동 테스트종료

    # 원본 mp4 의 cttmstsn 에 해당하는 summary, keyword 가져오기
    keyword = ",".join(find_keyword_id_in_db(cttmngno)).split(",")

    # keyword를 핵심키워드 와 이미지생성용 프롬프트로 이원화
    # LLMResult 직접 반영가능
    try:
        llmresultStr = event["LLMResult"]

    except Exception as e:
        print(f"not found event llm result - {e}")
        llmresultStr = find_LLMResult_id_in_db(cttmngno)

    if llmresultStr == None:
        llmresultStr = find_LLMResult_id_in_db(cttmngno)

    print("-------------------------")
    print(llmresultStr)
    print("-------------------------")
    llmJson = json.loads(llmresultStr)

    scenes = llmJson["movieInfo"]["scenes"]

    # summary = find_summary_in_db(cttmngno)
    # ctt_title = find_ctt_title_in_db(cttmngno)
    # ctt_title = split_long_lines(ctt_title, 25)
    summary = llmJson["summary"]
    ctt_title = find_ctt_title_in_db(cttmngno)
    ctt_title = split_long_lines(ctt_title, 25)

    narrator = llmJson["movieInfo"]["narrator"]

    print("gen info", scenes, narrator)

    # summary -> polly (TTS)
    # audio_file_s3_key = generate_audio(summary, S3_BUCKET, f"temp_LXP/genShortForms/output/audio/{cttmngno}.mp3")

    # summary -> open ai(TTS)
    audio_file_s3_key = generate_and_upload_tts(
        summary, S3_BUCKET, f"temp_LXP/genShortForms/output/audio/{cttmngno}_1.mp3"
    )
    # audio_file_s3_key = f"temp_LXP/genShortForms/output/audio/{cttmngno}_1.mp3"

    print("audio_file_s3_key:", audio_file_s3_key)

    # open ai (STT)
    vtt_json = generate_summary_vtt(audio_file_s3_key)
    print("vtt_json", vtt_json)

    # DALL-E 이미지 생성 (keyword -> open ai api -> Dalle image생성)
    num_images = 1
    # keyword => 이미지 생성용
    # image_urls = genDalleImage(keyword, num_images)
    image_urls = genDalleImageWithPrompts(scenes, num_images)

    # image_urls =  ['https://oaidalleapiprodscus.blob.core.windows.net/private/org-xlYC2ZFL1sBnYgCcqEnzaxoZ/user-kxckg9ANqiA2O37glRC7APUG/img-BVFNwJwbDjtTAm2v1LD7FJdz.png?st=2025-02-04T07%3A46%3A21Z&se=2025-02-04T09%3A46%3A21Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-02-04T00%3A36%3A02Z&ske=2025-02-05T00%3A36%3A02Z&sks=b&skv=2024-08-04&sig=fPPhsvoRSVKfESSBY7zVxvA3uijBqD7tyEdKR7meMEU%3D', 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-xlYC2ZFL1sBnYgCcqEnzaxoZ/user-kxckg9ANqiA2O37glRC7APUG/img-lrWOFqZvlpF6Bmt2z3eG1EjI.png?st=2025-02-04T07%3A46%3A35Z&se=2025-02-04T09%3A46%3A35Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-02-04T00%3A25%3A23Z&ske=2025-02-05T00%3A25%3A23Z&sks=b&skv=2024-08-04&sig=jpf4sumJjACf4zbdNXklCFk9iDBrjebPq0Rz03tSNYg%3D', 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-xlYC2ZFL1sBnYgCcqEnzaxoZ/user-kxckg9ANqiA2O37glRC7APUG/img-W8VbZ1Gw9mxft5HM65x9F4qe.png?st=2025-02-04T07%3A46%3A45Z&se=2025-02-04T09%3A46%3A45Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-02-04T00%3A24%3A25Z&ske=2025-02-05T00%3A24%3A25Z&sks=b&skv=2024-08-04&sig=WGHf54PqIRcL5YJzgFeP%2BSBUYBtL3plpVoLPnnMnA%2BE%3D', 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-xlYC2ZFL1sBnYgCcqEnzaxoZ/user-kxckg9ANqiA2O37glRC7APUG/img-1I1RNwcSdb76eJVhgRPtsbqz.png?st=2025-02-04T07%3A47%3A10Z&se=2025-02-04T09%3A47%3A10Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-02-04T00%3A26%3A06Z&ske=2025-02-05T00%3A26%3A06Z&sks=b&skv=2024-08-04&sig=M8pW66ITt0L5jTSGXsSmcpGJ376wqGsCd2N2esbSKek%3D', 'https://oaidalleapiprodscus.blob.core.windows.net/private/org-xlYC2ZFL1sBnYgCcqEnzaxoZ/user-kxckg9ANqiA2O37glRC7APUG/img-iVrLXvgPjhZiLn7PHkVl2U7X.png?st=2025-02-04T07%3A47%3A19Z&se=2025-02-04T09%3A47%3A19Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-02-04T00%3A17%3A50Z&ske=2025-02-05T00%3A17%3A50Z&sks=b&skv=2024-08-04&sig=YmXqnwnIYly%2B4bbHWplVCC8bx9iFmnLS5ZXLrQyBP/4%3D']
    print("image_urls:", image_urls)

    # download_and_upload_to_s3(image_urls, cttmngno)

    if not image_urls:
        return {"statusCode": 500, "body": "Failed to generate images"}

    # 비디오 출력 경로 (경로 중복 방지)
    output_file = f"/tmp/generated_video_{cttmngno}_{int(time.time())}.mp4"

    # 비디오 생성
    video_path = makeShortformWithImages(
        image_urls, ctt_title, cttmngno, audio_file_s3_key, output_file, vtt_json
    )

    if video_path:
        print("videopath", video_path)

        # # S3 업로드
        # s3_output_key = f"temp_LXP/genShortForms/output/videos/generated_video_{int(time.time())}.mp4"
        # s3_client.upload_file(video_path, S3_BUCKET, s3_output_key)
        # s3_output_key = f"temp_LXP/genShortForms/output/videos/generated_video_{int(time.time())}.mp4"

        # S3 업로드 는 기존 ctt_path에 저장하고 CMS_CTT_TRANS에 기록한다
        ctt_path = find_cttpath_in_db(cttmngno)
        fin_upload_path = os.path.dirname(ctt_path)

        s3_output_key = f"{fin_upload_path}/generated_video_{int(time.time())}.mp4"
        s3_client.upload_file(video_path, S3_BUCKET, s3_output_key)

        print(f"final mp4 file: https://devlxpcms.kbstar.com/{s3_output_key}")

        # DB에 기록
        update_ctt_path_in_db(cttmngno, s3_output_key)

        message = {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Video successfully generated and uploaded",
                    "video_s3_path": f"s3://{S3_BUCKET}/{s3_output_key}",
                }
            ),
        }

        # 작업 완료시 SNS 전송
        sns = boto3.client("sns")
        resp_sns = sns.publish(
            TopicArn=os.environ.get("SNS_ARN_JOBFINISH"),
            Message=json.dumps(message),
            Subject="task Completion GEN AI shorts",
        )

        print("SNS publish resp", resp_sns)
        # json return
        return message

    else:
        return {"statusCode": 500, "body": "Failed to create video"}
