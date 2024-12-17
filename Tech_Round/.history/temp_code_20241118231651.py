import yt_dlp
import json

# Define the categories and the links to scrape
video_data = {
    "FEATURED": [
        "https://www.youtube.com/watch?v=tyIG2NwMJV4&t=27s",
        "https://www.youtube.com/watch?v=xulwgTRFZ-c",
        "https://www.youtube.com/watch?v=UbNcrinMPDc",
        "https://www.youtube.com/watch?v=m9OKN053ueM",
        "https://www.youtube.com/watch?v=eKsGXd5UXFk&t=13s",
        "https://www.youtube.com/watch?v=wgo9Ziq4EfA"
    ],
    "EducationL": [
        "https://www.youtube.com/watch?v=PgTxuo7SagI&t=101s",
        "https://www.youtube.com/watch?v=GwCxu8pn338",
        "https://www.youtube.com/watch?v=0rlU7kQwZ24",
        "https://www.youtube.com/watch?v=AMXxXoHtM-o&t=6s",
        "https://www.youtube.com/watch?v=9x7DozCqLxU",
        "https://www.youtube.com/watch?v=jpDRfaWYk3I&t=6s"
    ],
    "Companies":[
        "https://www.youtube.com/watch?v=ExYUN5bdSPY",
"https://www.youtube.com/watch?v=ReYyIFCY0XA",
"https://www.youtube.com/watch?v=0P-RaFg0D2o",
"https://www.youtube.com/watch?v=j9hz3WZo1yY",
"https://www.youtube.com/watch?v=jPkq1ZL3i38&t=14s",
"https://www.youtube.com/watch?v=9VTZPJNjHqY",
"https://www.youtube.com/watch?v=_J120_nqjKY",
"https://www.youtube.com/watch?v=iABzSim4Pfw",
"https://www.youtube.com/watch?v=5ZAiil5g1_o",
"https://www.youtube.com/watch?v=0cnP4uUF07g",
"https://www.youtube.com/watch?v=InQx_ca_lGc",
"https://www.youtube.com/watch?v=g9TT0p0SPSg",
"https://www.youtube.com/watch?v=E-EtZmoR2DU",
"https://www.youtube.com/watch?v=et0Nt0XVYqU",
"https://www.youtube.com/watch?v=e7k1WIAR-fE&list=PLnk_WM-WosgWq32DZLcOg4fwwRtS-BbtT&index=2",
"https://www.youtube.com/watch?v=jFTSRu_48s8",
"https://www.youtube.com/watch?v=YmtGrIEufXw",
"https://www.youtube.com/watch?v=nKoT-b4CCUY&list=PL_Nq1u5fxtkeF2B4HxbXySM0JRaipnkhB",
"https://www.youtube.com/watch?v=Cp3-riVNFUM&list=PL_Nq1u5fxtkeF2B4HxbXySM0JRaipnkhB&index=10",
"https://www.youtube.com/watch?v=UrglHfZfW6U",
"https://www.youtube.com/watch?v=MxKndyaQ-C0",
"https://www.youtube.com/watch?v=iz6X3cWcsJA",
"https://www.youtube.com/watch?v=jJCfNFkpDp8",
"https://www.youtube.com/watch?v=gsJs6YfU66s",
"https://www.youtube.com/watch?v=YAt9otHZzq0",
"https://www.youtube.com/watch?v=2RYAr-Sb9WQ",
"https://www.youtube.com/watch?v=7orqKQPOkfI",
"https://www.youtube.com/watch?v=T0vLvh4BGmQ",
"https://www.youtube.com/watch?v=hQ2Jv55qfng",
"https://www.youtube.com/watch?v=eXhPF9EF4w8",
"https://www.youtube.com/watch?v=nTLr5J_PYns",
"https://www.youtube.com/watch?v=ZEMZLU9gwDM",
"https://www.youtube.com/watch?v=rVlnwFvkbFM",
"https://www.youtube.com/watch?v=8I1ghbtXd_c",
"https://www.youtube.com/watch?v=no9uNMj0xU4",
"https://www.youtube.com/watch?v=AcI-A7uCNoA",
"https://www.youtube.com/watch?v=nJd4b5lutY0",
"https://www.youtube.com/watch?v=0cnP4uUF07g&t=3s",
"https://www.youtube.com/watch?v=vMbFna4iKv0",
"https://www.youtube.com/watch?v=nnGt55eh7SM",
"https://www.youtube.com/watch?v=lNyUuMEPRm4",
"https://www.youtube.com/watch?v=h63QhLTA5kk",
"https://www.youtube.com/watch?v=5H54aT8TyDo",
"https://www.youtube.com/watch?v=P3GnsSQU6hA",
"https://www.youtube.com/watch?v=MhORW8OxYSc",
"https://www.youtube.com/watch?v=Uj9GbxqwiPw",
"https://www.youtube.com/watch?v=WsEI74cF4Ug",
"https://www.youtube.com/watch?v=dJdFWvUQWBA",
"https://www.youtube.com/watch?v=QhsNsQlKz-w&t=25s",
"https://www.youtube.com/watch?v=wbgxGlWZ5sA",
"https://www.youtube.com/watch?v=PJ76RARMSj0",
"https://www.youtube.com/watch?v=gV0MhOzPy8M",
"https://www.youtube.com/watch?v=sSYSy2xOyqhc",
"https://www.youtube.com/watch?v=Fa0i2KouPok",
"https://www.youtube.com/watch?v=-NtegUkhml4",
"https://www.youtube.com/watch?v=xTI7gHxbnB0",
"https://www.youtube.com/watch?v=Kw87vGkiSUQ",
"https://www.youtube.com/watch?v=uMv9FsQvBPg&t=17s",
"https://www.youtube.com/watch?v=KX4pOTG-0Co",
"https://www.youtube.com/watch?v=ua5AoNDOPeg&list=PLnk_WM-WosgUEymST4wnEClEv1WDMM1zs&index=1",
"https://www.youtube.com/watch?v=ff6TumloT_4&list=PLXgwvWUNiAP-IfFfjpOvowxF2sLqFh7Xi",
"https://www.youtube.com/watch?v=Zu48OqjjJB4",
"https://www.youtube.com/watch?v=F3yIN8kRipk",
"https://www.youtube.com/watch?v=LFw1RC1S3Xc",
"https://www.youtube.com/watch?v=zrU91rGkJgY",
"https://www.youtube.com/watch?v=5ZMwg8wr4Kg",
"https://www.youtube.com/watch?v=IgZCt7Yvpxw",
"https://www.youtube.com/watch?v=iQkhAUtnDN0",
"https://www.youtube.com/watch?v=7P5UA4FxSQo",
"https://www.youtube.com/embed/Na-B_K5OCw8?rel=0&amp;autoplay=1",
"https://www.youtube.com/watch?v=m9OKN053ueM",
"https://www.youtube.com/watch?v=BnSNSS5zktY",
"https://www.youtube.com/watch?v=Jx8hSUvHDU0",
"https://www.youtube.com/watch?v=V9C8jDmicbg",
"https://www.youtube.com/watch?v=DyAa6-RSBBM",
"https://www.youtube.com/watch?v=q8-alDr1f7U"

    ]
}

# Function to extract video details with error handling
def extract_video_details(url):
    ydl_opts = {
        'quiet': True,
        'extract_flat': False,
        'force_generic_extractor': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            
            # Extract necessary details
            video_info = {
                "url": url,
                "channel_name": info_dict.get('uploader', 'Unknown'),
                "video_title": info_dict.get('title', 'No title available'),
                "category": info_dict.get('categories', ['No category available'])[0]
            }
            return video_info
    except yt_dlp.utils.ExtractorError as e:
        print(f"Error extracting {url}: {str(e)}")
        return None
    except yt_dlp.utils.DownloadError as e:
        print(f"Error with download: {str(e)}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred for {url}: {str(e)}")
        return None

# List to hold all the video details
all_video_details = []

# Loop through each category and its links, and scrape the data
for category, urls in video_data.items():
    for url in urls:
        video_details = extract_video_details(url)
        if video_details:
            video_details["category"] = category  # Set the category for each video
            all_video_details.append(video_details)

# Save the scraped data to a JSON file
with open('video_details.json', 'w', encoding='utf-8') as json_file:
    json.dump(all_video_details, json_file, ensure_ascii=False, indent=4)

print("Scraping complete. Video details saved to 'video_details.json'.")
