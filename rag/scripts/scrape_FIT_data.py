from bs4 import BeautifulSoup
import requests
import argparse
import os, sys 
import re
from urllib.parse import urljoin
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.common import load_yaml, save_json, load_json

parser = argparse.ArgumentParser(description="Scrape FIT data")
parser.add_argument("--config_path", type=str, default="data_configs/fit.yaml", help="Path to the YAML config file")
parser.add_argument("--output_folder", type=str, default="extracted_data", help="Folder to save the extracted JSON files")

from bs4 import BeautifulSoup

from bs4 import BeautifulSoup

def extract_course_tables(soup):
    curriculum = []
    
    # 1. Target all tables directly
    tables = soup.find_all('table')
    
    for table in tables:
        # 2. Extract the semester name from the caption > h4
        caption = table.find('caption')
        if not caption:
            continue # Skip if it's some other random UI table without a caption
            
        semester_header = caption.find(['h3', 'h4'])
        semester_name = semester_header.get_text(strip=True) if semester_header else "Unknown Semester"
        
        semester_data = {
            "semester": semester_name,
            "courses": []
        }
        
        # 3. Target the tbody specifically to naturally ignore the thead headers
        tbody = table.find('tbody')
        if not tbody:
            continue
            
        rows = tbody.find_all('tr')
        
        for row in rows:
            # 4. CRITICAL FIX: Find both <th> and <td> tags to handle the mixed row structure
            cols = row.find_all(['th', 'td'])
            
            # Make sure the row actually has the expected number of columns
            if len(cols) >= 6:
                # Extract the hyperlink from the Title column (index 1)
                link_tag = cols[1].find('a')
                course_url = link_tag['href'] if link_tag else None
                
                course = {
                    "abbreviation": cols[0].get_text(strip=True),
                    "title": cols[1].get_text(strip=True),
                    "credits": cols[2].get_text(strip=True),
                    "duty": cols[3].get_text(strip=True),
                    "completion": cols[4].get_text(strip=True),
                    "faculty": cols[5].get_text(strip=True),
                    "course_url": course_url
                }
                semester_data["courses"].append(course)
                
        # Only append if we actually found courses
        if semester_data["courses"]:
            curriculum.append(semester_data)
            
    return curriculum

def extract_narrative_sections(soup):
    sections = []
    
    # 1. Find all "left column" title cells
    title_cells = soup.find_all('div', class_=lambda x: x and 'size--t-4-12' in x)
    
    for title_cell in title_cells:
        # Clean up the title text (removes extra tabs/newlines)
        section_title = " ".join(title_cell.get_text(strip=True).split())
        
        # 2. Find the adjacent content cell
        content_cell = title_cell.find_next_sibling('div', class_=lambda x: x and 'size--t-8-12' in x)
        
        if content_cell and section_title:
            content_wrapper = content_cell.find('div', class_='b-detail__content')
            
            if content_wrapper:
                section_lines = []
                
                # Check for standard structured elements
                structured_elements = content_wrapper.find_all(['p', 'li'])
                
                if structured_elements:
                    # Method A: Process lists and paragraphs
                    for el in structured_elements:
                        text = " ".join(el.get_text(strip=True).split())
                        if text:
                            if el.name == 'li':
                                section_lines.append(f"- {text}")
                            else:
                                section_lines.append(text)
                else:
                    # Method B (THE FIX): Fallback for raw text, <a> tags, or weird divs
                    # We use a separator so "Herout Adam chairman" becomes "Herout Adam | chairman"
                    raw_text = content_wrapper.get_text(separator=' | ', strip=True)
                    if raw_text:
                        section_lines.append(raw_text)
                
                section_text = "\n".join(section_lines)
                
                if section_text:
                    sections.append({
                        "section_title": section_title,
                        "content": section_text
                    })
                    
    return sections

def scrape_program_data(config, save_folder):
    for item in config:
        save_dict = {}
        name = item['name']
        url = item['url']
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        summary_container = soup.find('div', class_='b-detail__summary')
        paragraphs = summary_container.find_all('p', class_='mb10')
        metadata = {}
        for p in paragraphs:
            strong_tag = p.find('strong')
            if strong_tag:
                value = strong_tag.text.strip()
                raw_label = p.text.replace(value, '').strip()
                clean_label = raw_label.rstrip(':').strip().lower().replace(' ', '_')
                metadata[clean_label] = value
        save_dict["metadata"] = metadata
        heading_container = soup.find('div', class_='b-detail__head')
        main_heading = heading_container.find('p', class_='mb20').text.strip()
        program_heading = heading_container.find('h1', class_='b-detail__title').text.strip()
        save_dict['main_heading'] = main_heading  
        save_dict['program_heading'] = program_heading
        
        narrative_sections = extract_narrative_sections(soup)
        metadata = {}
        for section in narrative_sections:
            metadata[f"{section['section_title'].lower().replace(' ', '_')}"] = section['content']
        save_dict["program_details"] = metadata
        
        curriculum = extract_course_tables(soup)
        save_dict["curriculum"] = curriculum

        save_path = os.path.join(save_folder, f"{name}.json")
        save_json(save_dict, save_path)

from bs4 import BeautifulSoup

def extract_course_header(soup):
    course_data = {
        "metadata": {},
        "content": ""
    }
    
    # 1. Extract Metadata using the wrapper class
    annot_paragraph = soup.find('p', class_='b-detail__annot')
    
    if annot_paragraph:
        # Find all the individual spans holding the data
        spans = annot_paragraph.find_all('span', class_='b-detail__annot-item')
        
        # We can map them by index since the visual order is standard
        # [Code] - [Year] - [Semester] - [Credits]
        if len(spans) >= 1:
            # Using itemprop for extra safety on the course code
            code_span = annot_paragraph.find('span', attrs={"itemprop": "courseCode"})
            course_data["metadata"]["code"] = code_span.get_text(strip=True) if code_span else spans[0].get_text(strip=True)
            
        if len(spans) >= 2:
            course_data["metadata"]["academic_year"] = spans[1].get_text(strip=True)
            
        if len(spans) >= 3:
            course_data["metadata"]["semester"] = spans[2].get_text(strip=True)
            
        if len(spans) >= 4:
            course_data["metadata"]["credits"] = spans[3].get_text(strip=True)

    # 2. Extract the Description/Abstract
    # Using the itemprop attribute directly is the most robust targeting method
    abstract_div = soup.find('div', attrs={"itemprop": "description"})
    
    if abstract_div:
        # Reusing our robust markdown-list logic from earlier
        content_elements = abstract_div.find_all(['p', 'li'])
        
        description_lines = []
        for el in content_elements:
            # Clean up the text
            text = " ".join(el.get_text(strip=True).split())
            if text:
                # Add markdown bullets for list items
                if el.name == 'li':
                    description_lines.append(f"- {text}")
                else:
                    description_lines.append(text)
                    
        # Join with double newlines for clear paragraph separation in the LLM context
        course_data["content"] = "\n\n".join(description_lines)
        
    return course_data

from bs4 import BeautifulSoup

def extract_personnel_links(soup):
    # Dictionary to hold our final JSON structure
    personnel_data = {}
    
    # The keywords we are looking for in the left column titles
    target_roles = ['guarantor', 'coordinator', 'lecturer', 'instructor']
    
    # 1. Find all "left column" title cells
    title_cells = soup.find_all('div', class_=lambda x: x and 'size--t-4-12' in x)
    
    for title_cell in title_cells:
        section_title = " ".join(title_cell.get_text(strip=True).split())
        title_lower = section_title.lower()
        
        # 2. Check if this section matches any of our target roles
        if any(role in title_lower for role in target_roles):
            
            # Find the adjacent right column content cell
            content_cell = title_cell.find_next_sibling('div', class_=lambda x: x and 'size--t-8-12' in x)
            
            if content_cell:
                # 3. Extract all anchor tags within this content block
                links = content_cell.find_all('a')
                personnel_list = []
                
                for link in links:
                    name = " ".join(link.get_text(strip=True).split())
                    href = link.get('href')
                    
                    # Ensure it actually has an href and isn't just an empty anchor
                    if name and href:
                        # Optional: You can filter here to ensure it's a profile link
                        # if '/person/' in href: 
                        personnel_list.append({
                            "name": name,
                            "url": href
                        })
                
                # 4. If we found people, add them to the dictionary
                if personnel_list:
                    # Clean up the key name (e.g., "PROGRAMME GUARANTOR" -> "programme_guarantor")
                    key_name = section_title.replace(' ', '_').lower()
                    personnel_data[key_name] = personnel_list
                    
    return personnel_data
    
def scrape_course_data(config, save_folder):
    saved_programs = []
    accumulated_personnel_links = {}
    for item in config:
        name = item['name']
        saved_programs.append(os.path.join(save_folder, f"{name}.json"))
    save_folder = os.path.join(save_folder, "courses")
    os.makedirs(save_folder, exist_ok=True)
    for program_path in saved_programs:
        program_data = load_json(program_path)
        # Process the program data as needed
        # For example, you might want to extract course information
        courses = program_data.get("curriculum", [])
        for course_entry in courses:
            # Do something with each course
            for course in course_entry.get("courses", []):
                course_url = course.get("course_url")
                if course_url is None:
                    continue
                course_save_dict = {}
                response = requests.get(course_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                course_header = extract_course_header(soup)
                course_narrative = extract_narrative_sections(soup)
                personal_links = extract_personnel_links(soup)
                course_save_dict["metadata"] = course_header.get("metadata", {})
                m = {}
                for section in course_narrative:
                    m[f"{section['section_title'].lower().replace(' ', '_')}"] = section['content']
                course_save_dict["content"] = m
                course_code = course_save_dict["metadata"].get("code", "unknown_code")
                save_path = os.path.join(save_folder, f"{course_code}.json")
                save_json(course_save_dict, save_path)
                for link_type in personal_links:
                    for entry in personal_links[link_type]:
                        name = entry["name"]
                        url = entry["url"]
                        if name not in accumulated_personnel_links:
                            accumulated_personnel_links[name] = url
                        else:
                            # Optional: You can check for URL consistency here
                            if accumulated_personnel_links[name] != url:
                                print(f"Warning: Name '{name}' has conflicting URLs: '{accumulated_personnel_links[name]}' vs '{url}'")
    # After processing all courses, save the accumulated personnel links
    unique_urls = set(accumulated_personnel_links.values())
    print(f"Total unique personnel: {len(accumulated_personnel_links)}, Total unique URLs: {len(unique_urls)}")
    personnel_save_path = os.path.join(save_folder, "accumulated_personnel_links.json")
    save_json(accumulated_personnel_links, personnel_save_path)
                # exit()
from bs4 import BeautifulSoup

def extract_profile_tabs(soup, current_page_url):
    profile_urls = {}
    
    # 1. NEW: Target the specific profile navigation wrapper first!
    # This acts as a fence, ignoring the global site header.
    profile_nav = soup.find('div', class_=lambda x: x and 'b-profile__nav' in x)
    
    if not profile_nav:
        print("Could not find the profile navigation container.")
        return profile_urls
        
    # 2. Find the list items ONLY inside this specific profile nav fence
    list_items = profile_nav.find_all('li', class_=lambda x: x and 'm-main__item' in x)
    
    for li in list_items:
        link = li.find('a')
        if link:
            # We use string manipulation to clean out the "::before" pseudo-elements
            # that sometimes render as raw text in BeautifulSoup
            raw_name = link.get_text(strip=True)
            tab_key = raw_name.lower().replace(' ', '_').replace('\n', '')
            
            raw_href = link.get('href')
            
            # Optional: Ignore empty anchors or the Czech language toggle (usually '.cs')
            # if tab_key and href and href != '#' and not href.endswith('.cs'):
            #     profile_urls[tab_key] = href
            if tab_key and raw_href and raw_href != '#':
                # 2. THE FIX: Safely merge the base URL with the raw href
                # 'current_page_url' should be the URL of the page you just scraped
                # e.g., "https://www.fit.vut.cz"
                absolute_url = urljoin(current_page_url, raw_href)
                
                profile_urls[tab_key] = absolute_url
                
    return profile_urls

# --- Example Output ---
# {
#   "contact": "https://www.fit.vut.cz/person/meduna/.en",
#   "roles": "https://www.fit.vut.cz/person/meduna/functions/.en#nav",
#   "curriculum": "https://www.fit.vut.cz/person/meduna/curriculum/.en#nav",
#   "teaching": "https://www.fit.vut.cz/person/meduna/teaching/.en#nav"
# }


def extract_personel_roles(soup):
    roles_data = []
    
    # 1. Target the roles tab panel specifically
    roles_panel = soup.find('div', id='functions')
    
    if not roles_panel:
        return roles_data
        
    # 2. Target the main content column
    content_cell = roles_panel.find('div', class_=lambda x: x and 'size--t-8-12' in x)
    
    if not content_cell:
        return roles_data

    # 3. Use the <h3> tags as our primary anchors for each block
    headers = content_cell.find_all('h3')
    
    for h3 in headers:
        entry = {
            "organization": h3.get_text(strip=True),
            "role": None,
            "url": None
        }
        
        # 4. The role is usually in the immediate next sibling div with the 'fz-lg' class
        role_div = h3.find_next_sibling('div', class_=lambda x: x and 'fz-lg' in x)
        
        if role_div:
            entry["role"] = role_div.get_text(strip=True)
            
            # 5. The link is typically in the div right after the role div
            potential_url_div = role_div.find_next_sibling('div')
            
            # CRITICAL SAFETY CHECK: Ensure we haven't accidentally skipped to the next <h3> block
            # If a role doesn't have a link (like the 'Scientific Board' in your image), 
            # we don't want to accidentally grab the link from the NEXT role.
            if potential_url_div and potential_url_div.find_previous_sibling('h3') == h3:
                link = potential_url_div.find('a')
                if link:
                    entry["url"] = link.get('href')
                    
        roles_data.append(entry)
        
    return roles_data

# --- Example Output ---
# [
#   {
#     "organization": "Formal Model Research Group",
#     "role": "Principal researcher",
#     "url": "https://www.fit.vut.cz/research/group/fm/.en"
#   },
#   {
#     "organization": "Library Board",
#     "role": "Chairman",
#     "url": "https://www.fit.vut.cz/fit/kr/.en"
#   },
#   {
#     "organization": "Members of Commissions for State Final Examinations...",
#     "role": "chairman",
#     "url": null
#   }
# ]

def extract_personel_contact(soup):
    contact_data = {
        "department": None,
        "department_url": None,
        "department_abbr": None,
        "role": None,
        "details": {}
    }
    
    # 1. Target the specific contact tab panel
    contact_panel = soup.find('div', id='contact')
    
    if not contact_panel:
        return contact_data
        
    # 2. Target the main content column
    content_cell = contact_panel.find('div', class_=lambda x: x and 'size--t-8-12' in x)
    
    if not content_cell:
        return contact_data

    # 3. Extract Department info from the h3 tag
    dept_header = content_cell.find('h3')
    if dept_header:
        dept_link = dept_header.find('a')
        if dept_link:
            contact_data["department"] = dept_link.get_text(strip=True)
            contact_data["department_url"] = dept_link.get('href')

    # 4. Extract extra paragraphs (Abbreviation and Role)
    # The screenshot shows these use the 'fz-lg' class
    paragraphs = content_cell.find_all('p', class_=lambda x: x and 'fz-lg' in x)
    if len(paragraphs) >= 1:
        contact_data["department_abbr"] = paragraphs[0].get_text(strip=True)
    if len(paragraphs) >= 2:
        contact_data["role"] = paragraphs[1].get_text(strip=True)

    # 5. Extract the contact details table
    table = content_cell.find('table', class_='table-blank')
    if table:
        rows = table.find_all('tr')
        for row in rows:
            cols = row.find_all(['th', 'td'])
            
            if len(cols) == 2:
                # Clean up the key (e.g., "Work Phone" -> "work_phone")
                raw_key = cols[0].get_text(strip=True)
                key = raw_key.lower().replace(' ', '_').replace('-', '_')
                
                val_cell = cols[1]
                val_text = val_cell.get_text(strip=True)
                val_link = val_cell.find('a')
                
                # We store both the raw text and the href (useful for mailtos, tel:, or ORCID links)
                contact_data["details"][key] = {
                    "value": val_text,
                    "url": val_link.get('href') if val_link else None
                }
                
    return contact_data


from bs4 import BeautifulSoup

def extract_personel_curriculum(soup):
    curriculum_data = {}
    
    # 1. Target the curriculum tab panel
    cur_panel = soup.find('div', id='curriculum')
    
    if not cur_panel:
        return curriculum_data
        
    # 2. Target the main content column
    content_cell = cur_panel.find('div', class_=lambda x: x and 'size--t-8-12' in x)
    
    if not content_cell:
        return curriculum_data

    # 3. Use <h3> tags as dynamic section anchors
    headers = content_cell.find_all('h3')
    
    for h3 in headers:
        section_title = h3.get_text(strip=True)
        # Clean key: "Education and academic qualification" -> "education_and_academic_qualification"
        section_key = section_title.lower().replace(' ', '_')
        curriculum_data[section_key] = []
        
        # 4. Walk through all siblings until we hit the next <h3>
        curr_node = h3.find_next_sibling()
        
        while curr_node and curr_node.name != 'h3':
            
            # SCENARIO A: The Timeline List (Education)
            if curr_node.name == 'ul' and 'list-timeline' in curr_node.get('class', []):
                items = curr_node.find_all('li', class_='list-timeline__item')
                for li in items:
                    # Extract the date specifically
                    date_p = li.find('p', class_=lambda x: x and 'list-timeline__date' in x)
                    date_text = date_p.get_text(strip=True) if date_p else ""
                    
                    # Extract all other paragraphs (the institution and degree)
                    details = []
                    for p in li.find_all('p'):
                        if p != date_p:
                            details.append(p.get_text(strip=True))
                            
                    curriculum_data[section_key].append({
                        "date": date_text,
                        "details": " | ".join(details) # Flatten to e.g., "Palacky University | B.C."
                    })
            
            # SCENARIO B: Standard Lists or Sections (Scientific Activities, Awards)
            elif curr_node.name in ['ul', 'section']:
                items = curr_node.find_all('li')
                
                # If it's a bulleted list
                if items:
                    for li in items:
                        link = li.find('a')
                        text = " ".join(li.get_text(strip=True).split())
                        
                        entry = {"text": text}
                        if link:
                            entry["url"] = link.get('href')
                        curriculum_data[section_key].append(entry)
                
                # If it's just raw paragraphs inside a section
                else:
                    paragraphs = curr_node.find_all('p')
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if text:
                            curriculum_data[section_key].append({"text": text})
                            
            # Move to the next sibling element to continue the while loop
            curr_node = curr_node.find_next_sibling()
            
    return curriculum_data

from bs4 import BeautifulSoup

def extract_personel_teaching(soup):
    teaching_data = {}
    
    # 1. Target the teaching tab panel specifically
    teaching_panel = soup.find('div', id='teaching')
    
    if not teaching_panel:
        return teaching_data
        
    # 2. Target the main content column
    content_cell = teaching_panel.find('div', class_=lambda x: x and 'size--t-8-12' in x)
    
    if not content_cell:
        return teaching_data

    # 3. Use <h3> tags as section anchors
    headers = content_cell.find_all('h3')
    
    for h3 in headers:
        section_name = h3.get_text(strip=True)
        # e.g., "Guaranteed courses" -> "guaranteed_courses"
        section_key = section_name.lower().replace(' ', '_')
        
        # 4. Find the immediate sibling to parse
        # Consulting hours uses a <section>, courses use a <table>
        sibling = h3.find_next_sibling(['section', 'table'])
        
        if not sibling:
            continue
            
        # --- SCENARIO A: Consulting Hours ---
        if section_key == 'consulting_hours':
            # Extract the raw text and any links (like mailto:)
            text = sibling.get_text(separator=' ', strip=True)
            links = [{"text": a.get_text(strip=True), "url": a.get('href')} for a in sibling.find_all('a')]
            
            teaching_data[section_key] = {
                "text": text,
                "links": links
            }
            
        # --- SCENARIO B: Course Tables ---
        elif sibling.name == 'table':
            teaching_data[section_key] = []
            rows = sibling.find_all('tr')
            
            for row in rows:
                tds = row.find_all('td')
                if len(tds) >= 2:
                    course_abbr = tds[0].get_text(strip=True)
                    
                    # Target the primary course link
                    title_link = tds[1].find('a')
                    course_url = title_link.get('href') if title_link else None
                    
                    # Using stripped_strings extracts ['Course Name', 'English, winter,', 'DIFS']
                    text_parts = list(tds[1].stripped_strings)
                    
                    course_title = text_parts[0] if len(text_parts) > 0 else ""
                    
                    # Clean up the trailing comma on the meta string
                    course_meta = text_parts[1].strip(', ') if len(text_parts) > 1 else ""
                    
                    # The department is usually the last element
                    course_dept = text_parts[2] if len(text_parts) > 2 else ""
                    
                    teaching_data[section_key].append({
                        "abbreviation": course_abbr,
                        "title": course_title,
                        "details": course_meta,
                        "department": course_dept,
                        "url": course_url
                    })
                    
    return teaching_data

from bs4 import BeautifulSoup

def extract_personel_groupprojects(soup):
    projects_data = []

    # Find the b-profile__s-split section whose left column heading mentions "project"
    sections = soup.find_all('div', class_=lambda x: x and 'b-profile__s-split' in x)

    timeline = None
    for section in sections:
        heading = section.find(['h1', 'h2', 'h3'])
        if heading and 'project' in heading.get_text(strip=True).lower():
            timeline = section.find('ul', class_='list-timeline')
            break

    if not timeline:
        return projects_data

    items = timeline.find_all('li', class_='list-timeline__item')

    for li in items:
        date_p = li.find('p', class_=lambda x: x and 'list-timeline__date' in x)
        year = date_p.get_text(strip=True) if date_p else "Unknown Year"

        project_paragraphs = li.find_all('p')

        for p in project_paragraphs:
            if p == date_p:
                continue

            project_entry = {"year": year, "title": None, "url": None, "details": None}

            link = p.find('a')
            title_text = ""

            if link:
                title_text = link.get_text(strip=True)
                project_entry["title"] = title_text
                project_entry["url"] = link.get('href')

            full_text = p.get_text(separator=" ", strip=True)

            if title_text and full_text.startswith(title_text):
                metadata = full_text[len(title_text):].lstrip(', -').strip()
            else:
                metadata = full_text

            project_entry["details"] = metadata
            projects_data.append(project_entry)

    return projects_data

def extract_personel_projects(soup):
    projects_data = []
    
    # 1. Target the projects tab panel
    projects_panel = soup.find('div', id='projects')
    
    if not projects_panel:
        return projects_data
        
    # 2. Target the timeline list directly
    timeline = projects_panel.find('ul', class_='list-timeline')
    
    if not timeline:
        return projects_data

    # 3. Iterate through each timeline item (the year clusters)
    items = timeline.find_all('li', class_='list-timeline__item')
    
    for li in items:
        # Extract the year/date
        date_p = li.find('p', class_=lambda x: x and 'list-timeline__date' in x)
        year = date_p.get_text(strip=True) if date_p else "Unknown Year"
        
        # Find all other paragraphs inside this list item (the actual projects)
        project_paragraphs = li.find_all('p')
        
        for p in project_paragraphs:
            # Skip the date paragraph we already processed
            if p == date_p:
                continue
                
            project_entry = {
                "year": year,
                "title": None,
                "url": None,
                "details": None
            }
            
            # 4. Extract the title and URL
            link = p.find('a')
            title_text = ""
            
            if link:
                title_text = link.get_text(strip=True)
                project_entry["title"] = title_text
                project_entry["url"] = link.get('href')
                
            # 5. Extract the metadata cleanly (funding, dates, status)
            # We get the full text and slice out the title to leave just the metadata
            full_text = p.get_text(separator=" ", strip=True)
            
            if title_text and full_text.startswith(title_text):
                # Remove the title and clean up any leftover leading commas or spaces
                metadata = full_text[len(title_text):].lstrip(', -').strip()
            else:
                metadata = full_text
                
            project_entry["details"] = metadata
            
            projects_data.append(project_entry)
            
    return projects_data

from bs4 import BeautifulSoup

def extract_personel_publicationresults(soup):
    publications_data = []
    
    # 1. Target the publication results tab panel
    pub_panel = soup.find('div', id='publication-results')
    
    if not pub_panel:
        return publications_data
        
    # 2. Target the timeline list
    timeline = pub_panel.find('ul', class_='list-timeline')
    
    if not timeline:
        return publications_data

    # 3. Iterate through each timeline item (the year clusters)
    items = timeline.find_all('li', class_='list-timeline__item')
    
    for li in items:
        # Extract the year
        date_p = li.find('p', class_=lambda x: x and 'list-timeline__date' in x)
        year = date_p.get_text(strip=True) if date_p else "Unknown Year"
        
        # Find all other paragraphs (the actual publications)
        pub_paragraphs = li.find_all('p')
        
        for p in pub_paragraphs:
            # Skip the date paragraph we already processed
            if p == date_p:
                continue
                
            # 4. Extract the "Detail" link URL
            link = p.find('a', class_=lambda x: x and 'list-links__link' in x)
            # Fallback just in case they drop the class name
            if not link:
                 link = p.find('a', string="Detail")
                 
            pub_url = link.get('href') if link else None
            
            # 5. Extract the full citation text
            citation_text = p.get_text(separator=" ", strip=True)
            
            # Clean up the trailing "Detail" text so it doesn't pollute the LLM context
            if citation_text.endswith('Detail'):
                citation_text = citation_text[:-6].strip()
                
            publications_data.append({
                "year": year,
                "citation": citation_text,
                "url": pub_url
            })
            
    return publications_data


from bs4 import BeautifulSoup

def extract_personel_otherresults(soup):
    other_results_data = []
    
    # 1. Target the other results tab panel
    results_panel = soup.find('div', id='other-results')
    
    if not results_panel:
        return other_results_data
        
    # 2. Target the timeline list
    timeline = results_panel.find('ul', class_='list-timeline')
    
    if not timeline:
        return other_results_data

    # 3. Iterate through each timeline item (the year clusters)
    items = timeline.find_all('li', class_='list-timeline__item')
    
    for li in items:
        # Extract the year
        date_p = li.find('p', class_=lambda x: x and 'list-timeline__date' in x)
        year = date_p.get_text(strip=True) if date_p else "Unknown Year"
        
        # Find all other paragraphs (the actual items)
        item_paragraphs = li.find_all('p')
        
        for p in item_paragraphs:
            # Skip the date paragraph
            if p == date_p:
                continue
                
            # 4. Extract Title and URL
            link = p.find('a')
            title = link.get_text(strip=True) if link else "Unknown Title"
            url = link.get('href') if link else None
            
            # 5. Extract Metadata and Authors using stripped_strings
            # This generates a list like: ['Title', ', biography, 2024', 'Authors', ': MEDUNA, A.']
            strings = list(p.stripped_strings)
            
            metadata = ""
            authors = ""
            
            if len(strings) > 1:
                # Join everything after the title into a single string
                raw_details = " ".join(strings[1:])
                
                # Split the string cleanly if the 'Authors' tag is present
                if 'Authors' in raw_details:
                    parts = raw_details.split('Authors')
                    # Clean up trailing commas/spaces from the metadata part
                    metadata = parts[0].strip(', ').strip()
                    # Clean up leading colons/spaces from the authors part
                    authors = parts[1].strip(': ').strip()
                else:
                    # Fallback if there are no authors listed
                    metadata = raw_details.strip(', ').strip()
            
            other_results_data.append({
                "year": year,
                "title": title,
                "metadata": metadata,
                "authors": authors,
                "url": url
            })
            
    return other_results_data

# --- Example Output ---
# [
#   {
#     "year": "2024",
#     "title": "Brněnský rodák Kurt Gödel a jeho odkaz...",
#     "metadata": "biography, 2024",
#     "authors": "MEDUNA, A.",
#     "url": "https://www.fit.vut.cz/research/result/..."
#   },
#   {
#     "year": "2007",
#     "title": "Descriptional Complexity of Multigrammars: An Overview",
#     "metadata": "other unclassified results, 2007",
#     "authors": "MEDUNA, A.; LUKÁŠ, R.; FIALA, J.",
#     "url": "https://www.fit.vut.cz/research/result/..."
#   }
# ]

def scrape_personnel_links(save_folder):
    path = os.path.join(save_folder, "accumulated_personnel_links.json")
    if not os.path.exists(path):
        print(f"Error: Personnel links file not found at {path}. Please run the course data scraping first to accumulate personnel links.")
        return
    personnel_links = load_json(path)
    save_folder = os.path.join(save_folder, "personnel_profiles")
    os.makedirs(save_folder, exist_ok=True)
    for name, url in tqdm(personnel_links.items(), desc="Scraping Personnel Profiles"):
        save_path = os.path.join(save_folder, f"{name.replace(' ', '_')}.json")
        if os.path.exists(save_path):
            print(f"Profile for '{name}' already exists. Skipping...")
            continue
        response = requests.get(url)
        true_base_url = response.url
        soup = BeautifulSoup(response.content, 'html.parser')
        profile_tabs = extract_profile_tabs(soup, true_base_url)
        save_dict = {}
        for tab_key, tab_url in profile_tabs.items():
            tab_response = requests.get(tab_url)
            tab_soup = BeautifulSoup(tab_response.content, 'html.parser')
            extract_fn = f"extract_personel_{tab_key}"
            if tab_key in ["appliedresults", "results_with_impacton_practice", "applied_results", "results_with_impact_on_practice", "action", "other"]:
                extract_fn = "extract_personel_otherresults"
            if tab_key in ["publication_results"]:
                extract_fn = "extract_personel_publicationresults"
            assert extract_fn in globals(), f"No extraction function defined for tab '{tab_key}' for {name}, {url}"
            tab_content = globals()[extract_fn](tab_soup)
            save_dict[tab_key] = tab_content
        save_dict["name"] = name
        save_dict["profile_url"] = url
        
        save_json(save_dict, save_path)         


def extract_group_profile_tabs(soup, current_page_url):
    profile_urls = {}
    
    # 1. UPGRADED FENCE: Target both People and Group templates
    # People use: <div class="b-profile__nav">
    # Groups use: <nav class="m-main--sub">
    profile_nav = soup.find(['nav', 'div'], class_=lambda x: x and ('b-profile__nav' in x or 'm-main--sub' in x))
    
    if not profile_nav:
        print(f"Could not find the navigation container on {current_page_url}")
        return profile_urls
        
    # 2. Find the list items ONLY inside this specific fence
    list_items = profile_nav.find_all('li', class_=lambda x: x and 'm-main__item' in x)
    
    for li in list_items:
        link = li.find('a')
        if link:
            raw_name = link.get_text(strip=True)
            tab_key = raw_name.lower().replace(' ', '_').replace('\n', '')
            
            raw_href = link.get('href')
            
            if tab_key and raw_href and raw_href != '#':
                # Safely merge the base URL with the raw href
                absolute_url = urljoin(current_page_url, raw_href)
                profile_urls[tab_key] = absolute_url
                
    return profile_urls


from bs4 import BeautifulSoup

def extract_personel_about(soup):
    about_data = {}

    # ── NEW: capture the free-text intro block ──────────────────────────────
    # The first content area sits in a grid__cell that has b-detail__content
    # but lives inside a b-profile__s-split row that has NO h2/h3 heading.
    intro_cell = soup.find(
        'div',
        class_=lambda x: x and 'b-detail__content' in x and 'size--t-8-12' in x
    )
    if intro_cell:
        intro_paragraphs = []
        for p in intro_cell.find_all('p'):
            parts = p.decode_contents().split('<br>')
            for part in parts:
                text = BeautifulSoup(part, 'html.parser').get_text(strip=True)
                if text:
                    intro_paragraphs.append({"text": text})
        if intro_paragraphs:
            about_data["about"] = intro_paragraphs
    # ────────────────────────────────────────────────────────────────────────

    # Target all the split row containers directly
    rows = soup.find_all('div', class_=lambda x: x and 'b-profile__s-split' in x)

    for row in rows:
        title_cell = row.find('div', class_=lambda x: x and 'size--t-4-12' in x)
        if not title_cell:
            continue

        heading_tag = title_cell.find(['h2', 'h3'])
        if not heading_tag:
            continue

        section_title = heading_tag.get_text(strip=True)
        section_key = section_title.lower().replace(' ', '_')
        about_data[section_key] = []

        content_cell = row.find('div', class_=lambda x: x and 'size--t-8-12' in x)
        if content_cell:
            list_items = content_cell.find_all('li')

            if list_items:
                for li in list_items:
                    text = li.get_text(separator=" ", strip=True)
                    links = [{"text": a.get_text(strip=True), "url": a.get('href')} for a in li.find_all('a')]
                    entry = {"text": text}
                    if links:
                        entry["links"] = links
                    about_data[section_key].append(entry)

            else:
                paragraphs = content_cell.find_all('p')
                for p in paragraphs:
                    parts = p.decode_contents().split('<br>')
                    for part in parts:
                        text = BeautifulSoup(part, 'html.parser').get_text(strip=True)
                        if text:
                            about_data[section_key].append({"text": text})

    return about_data

def extract_personel_team(soup):
    team_data = {}

    sections = soup.find_all('div', class_=lambda x: x and 'b-profile__s-split' in x)

    for section in sections:
        heading_tag = section.find(['h1', 'h2', 'h3'])
        if not heading_tag:
            continue

        section_title = heading_tag.get_text(strip=True)
        section_key = section_title.lower().replace(' ', '_')

        members = []
        cards = section.find_all('div', class_='c-fit-group-members__item')

        for card in cards:
            link_tag = card.find('a', class_='c-fit-group-members__header')
            url = link_tag.get('href') if link_tag else None

            name_tag = card.find('h4', class_='c-fit-group-members__title')
            name = name_tag.get_text(strip=True) if name_tag else None

            position_tag = card.find('p', class_='c-fit-group-members__position')
            position = position_tag.get_text(strip=True) if position_tag else None

            notes = []
            text_div = card.find('div', class_='c-fit-group-members__text')
            if text_div:
                list_items = text_div.find_all('li')
                if list_items:
                    notes = [li.get_text(strip=True) for li in list_items]
                else:
                    raw = text_div.get_text(strip=True)
                    if raw:
                        notes = [raw]

            if name:
                entry = {"name": name, "position": position, "url": url}
                if notes:
                    entry["notes"] = notes
                members.append(entry)

        # ── THIS WAS MISSING ──────────────────────────
        if members:
            team_data[section_key] = members
        # ─────────────────────────────────────────────

    return team_data


def extract_timeline_section(soup, keyword="publication"):
    """
    Generic timeline extractor. Use keyword to target the right section.
    
    Examples:
        extract_timeline_section(soup, "project")
        extract_timeline_section(soup, "publication")
        extract_timeline_section(soup, "grant")
    """
    data = []

    sections = soup.find_all('div', class_=lambda x: x and 'b-profile__s-split' in x)

    timeline = None
    for section in sections:
        heading = section.find(['h1', 'h2', 'h3'])
        if heading and keyword.lower() in heading.get_text(strip=True).lower():
            timeline = section.find('ul', class_='list-timeline')
            break

    if not timeline:
        return data

    for li in timeline.find_all('li', class_='list-timeline__item'):
        date_p = li.find('p', class_=lambda x: x and 'list-timeline__date' in x)
        year = date_p.get_text(strip=True) if date_p else "Unknown Year"

        for p in li.find_all('p'):
            if p == date_p:
                continue

            entry = {"year": year, "title": None, "url": None, "details": None}

            link = p.find('a')
            title_text = ""
            if link:
                title_text = link.get_text(strip=True)
                entry["title"] = title_text
                entry["url"] = link.get('href')

            full_text = p.get_text(separator=" ", strip=True)
            if title_text and full_text.startswith(title_text):
                entry["details"] = full_text[len(title_text):].lstrip(', -').strip()
            else:
                entry["details"] = full_text

            data.append(entry)

    return data

def extract_applied_results(soup):
    data = []

    sections = soup.find_all('div', class_=lambda x: x and 'b-profile__s-split' in x)

    timeline = None
    for section in sections:
        heading = section.find(['h1', 'h2', 'h3'])
        if heading and 'applied' in heading.get_text(strip=True).lower():
            timeline = section.find('ul', class_='list-timeline')
            break

    if not timeline:
        return data

    for li in timeline.find_all('li', class_='list-timeline__item'):
        date_p = li.find('p', class_=lambda x: x and 'list-timeline__date' in x)
        year = date_p.get_text(strip=True) if date_p else "Unknown Year"

        for p in li.find_all('p'):
            if p == date_p:
                continue

            entry = {"year": year, "title": None, "url": None, "type": None, "authors": None, "extra": {}}

            # ── Title + URL from the bold link ────────────────────────────
            link = p.find('a', class_='list-links__link')
            if link:
                entry["title"] = link.get_text(strip=True)
                entry["url"] = link.get('href')

            # ── Parse <br>-separated segments for structured fields ───────
            # Render the inner HTML, split on <br>, parse each chunk
            raw_html = p.decode_contents()
            segments = raw_html.split('<br>')

            for segment in segments:
                seg_soup = BeautifulSoup(segment, 'html.parser')

                # Check if segment starts with a <strong> label, e.g. "Authors"
                strong = seg_soup.find('strong')
                if strong:
                    label = strong.get_text(strip=True).lower()  # e.g. "authors"
                    # The value is everything after the <strong> tag
                    strong.decompose()
                    value = seg_soup.get_text(separator=' ', strip=True).lstrip(':').strip()

                    if label == 'authors':
                        entry["authors"] = value
                    else:
                        entry["extra"][label] = value

                else:
                    # Plain text segment — likely type/category after the title
                    text = seg_soup.get_text(strip=True)
                    # Skip if it's just the title repeated
                    if text and entry["title"] and text.startswith(entry["title"]):
                        remainder = text[len(entry["title"]):].lstrip(', ').strip()
                        if remainder:
                            entry["type"] = remainder
                    elif text and text != entry["title"]:
                        if not entry["type"]:
                            entry["type"] = text

            data.append(entry)

    return data

def extract_results_with_impact(soup):
    data = []

    sections = soup.find_all('div', class_=lambda x: x and 'b-profile__s-split' in x)

    timeline = None
    for section in sections:
        heading = section.find(['h1', 'h2', 'h3'])
        if heading and 'impact' in heading.get_text(strip=True).lower():
            timeline = section.find('ul', class_='list-timeline')
            break

    if not timeline:
        return data

    for li in timeline.find_all('li', class_='list-timeline__item'):
        date_p = li.find('p', class_=lambda x: x and 'list-timeline__date' in x)
        year = date_p.get_text(strip=True) if date_p else "Unknown Year"

        for p in li.find_all('p'):
            if p == date_p:
                continue

            entry = {"year": year, "title": None, "url": None, "type": None, "authors": None}

            # ── Title + URL: the <a> contains everything before the <br> ──
            link = p.find('a', class_='list-links__link')
            if link:
                entry["url"] = link.get('href')

                # Split the link's inner HTML on <br> to separate title from type
                link_parts = link.decode_contents().split('<br>')
                if link_parts:
                    entry["title"] = BeautifulSoup(link_parts[0], 'html.parser').get_text(strip=True)
                if len(link_parts) > 1:
                    entry["type"] = BeautifulSoup(link_parts[1], 'html.parser').get_text(strip=True)

                # Remove the link so remaining text is just authors etc.
                link.decompose()

            # ── Authors from <strong> label ───────────────────────────────
            strong = p.find('strong')
            if strong:
                strong.decompose()  # remove the label itself
                remaining = p.get_text(separator=' ', strip=True).lstrip(':').strip()
                if remaining:
                    entry["authors"] = remaining

            data.append(entry)

    return data


def extract_other_results(soup):
    data = []

    sections = soup.find_all('div', class_=lambda x: x and 'b-profile__s-split' in x)

    timeline = None
    for section in sections:
        heading = section.find(['h1', 'h2', 'h3'])
        if heading and 'other' in heading.get_text(strip=True).lower():
            timeline = section.find('ul', class_='list-timeline')
            break

    if not timeline:
        return data

    for li in timeline.find_all('li', class_='list-timeline__item'):
        date_p = li.find('p', class_=lambda x: x and 'list-timeline__date' in x)
        year = date_p.get_text(strip=True) if date_p else "Unknown Year"

        for p in li.find_all('p'):
            if p == date_p:
                continue

            entry = {"year": year, "title": None, "url": None, "type": None, "authors": None, "extra": {}}

            link = p.find('a', class_='list-links__link')
            if link:
                entry["title"] = link.get_text(strip=True)
                entry["url"] = link.get('href')

            raw_html = p.decode_contents()
            segments = raw_html.split('<br>')

            for segment in segments:
                seg_soup = BeautifulSoup(segment, 'html.parser')
                strong = seg_soup.find('strong')
                if strong:
                    label = strong.get_text(strip=True).lower()
                    strong.decompose()
                    value = seg_soup.get_text(separator=' ', strip=True).lstrip(':').strip()
                    if label == 'authors':
                        entry["authors"] = value
                    else:
                        entry["extra"][label] = value
                else:
                    text = seg_soup.get_text(strip=True)
                    if text and entry["title"] and text.startswith(entry["title"]):
                        remainder = text[len(entry["title"]):].lstrip(', ').strip()
                        if remainder:
                            entry["type"] = remainder
                    elif text and text != entry["title"] and not entry["type"]:
                        entry["type"] = text

            data.append(entry)

    return data

def extract_equipment(soup):
    """
    Returns:
    {
        "lab_equipment": {
            "Network analyzers and testers": [
                {"name": "Spirent Test Center", "description": "equipment for testing..."},
                ...
            ],
            ...
        },
        "equipment_and_devices": {
            "category name": [...]
        }
    }
    """
    data = {}

    sections = soup.find_all('div', class_=lambda x: x and 'b-profile__s-split' in x)

    for section in sections:
        heading = section.find(['h1', 'h2', 'h3'])
        if not heading:
            continue

        heading_text = heading.get_text(strip=True)
        if 'equipment' not in heading_text.lower() and 'device' not in heading_text.lower():
            continue

        section_key = heading_text.lower().replace(' ', '_').replace('\xa0', '')
        content_cell = section.find('div', class_=lambda x: x and 'size--t-8-12' in x)
        if not content_cell:
            continue

        section_data = {}

        # Each category is a <p><b>Category name</b></p> followed by a <ul>
        children = list(content_cell.children)
        current_category = None

        for child in children:
            if child.name == 'p':
                b_tag = child.find('b')
                if b_tag:
                    current_category = b_tag.get_text(strip=True)
                    section_data[current_category] = []

            elif child.name == 'ul' and current_category:
                for li in child.find_all('li', recursive=False):
                    b_tag = li.find('b')
                    if b_tag:
                        name = b_tag.get_text(strip=True).rstrip(' -–')
                        b_tag.decompose()
                    else:
                        name = None

                    description = li.get_text(separator=' ', strip=True).lstrip(' -–').strip()

                    entry = {}
                    if name:
                        entry["name"] = name
                    if description:
                        entry["description"] = description

                    if entry:
                        section_data[current_category].append(entry)

        if section_data:
            data[section_key] = section_data

    return data

def scrape_group_data(config, save_folder):
    save_folder = os.path.join(save_folder, "groups")
    os.makedirs(save_folder, exist_ok=True)
    for item in config:
        name = item['name']
        url = item['url']
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        group_links = []
        link_list = soup.find('ul', class_='list-links')
        if not link_list:
            return group_links
            
        # 2. Iterate through the list items
        items = link_list.find_all('li', class_=lambda x: x and 'list-links__item' in x)
        for li in items:
            link = li.find('a')
            
            if link:
                group_name = link.get_text(strip=True)
                raw_href = link.get('href')
                
                # Ensure we have both a name and a link before saving
                if group_name and raw_href:
                    absolute_url = urljoin(url, raw_href)
                    
                    group_links.append({
                        "name": group_name,
                        "url": absolute_url
                    })
        for entry in tqdm(group_links):
            name, url = entry["name"], entry["url"]
            save_dict = {"group": name, "group_url": url}
            response = requests.get(url)
            true_base_url = response.url
            soup = BeautifulSoup(response.content, 'html.parser')
            profile_tabs = extract_group_profile_tabs(soup, true_base_url)
            for tab_key, tab_url in profile_tabs.items():
                tab_response = requests.get(tab_url)
                second_arg = None
                tab_soup = BeautifulSoup(tab_response.content, 'html.parser')
                extract_fn = f"extract_personel_{tab_key}"
                if tab_key in ["projects"]:
                    extract_fn = "extract_personel_groupprojects"
                if tab_key in ["publicationresults"]:
                    extract_fn = "extract_timeline_section"
                    second_arg = "publication"
                if tab_key in ["appliedresults", "applied_results", "results_with_impact_on_practice", "otherresults"]:
                    extract_fn = "extract_applied_results"
                    second_arg = None
                if tab_key in ["results_with_impact", "results_with_impacton_practice"]:
                    extract_fn = "extract_results_with_impact"
                if tab_key in ["otherresults"]:
                    extract_fn = "extract_other_results"
                if tab_key in ["equipment"]:
                    extract_fn = "extract_equipment"
                assert extract_fn in globals(), f"No extraction function defined for tab '{tab_key}' for {name}, {url}"
                
                if second_arg is None:
                    tab_content = globals()[extract_fn](tab_soup)
                else:
                    tab_content = globals()[extract_fn](tab_soup, second_arg)
                save_dict[tab_key] = tab_content
            save_path = os.path.join(save_folder, f"{name.replace(' ', '_')}.json")
            save_json(save_dict, save_path)


def normalize_key(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def sanitize_filename(text):
    cleaned = re.sub(r'[\\/:*?"<>|]+', '_', text).strip()
    cleaned = re.sub(r'\s+', '_', cleaned)
    return cleaned[:180]


def extract_project_header_metadata(soup):
    metadata = {}
    header = soup.find('div', class_=lambda x: x and 'b-detail__head' in x)
    if not header:
        return metadata

    annot_rows = header.find_all('p', class_=lambda x: x and 'b-detail__annot' in x)
    for row in annot_rows:
        text = " ".join(row.get_text(" ", strip=True).split())
        if not text:
            continue

        if ':' in text:
            label, value = text.split(':', 1)
            key = normalize_key(label)
            metadata[key] = value.strip()
        else:
            metadata[f"item_{len(metadata) + 1}"] = text

    return metadata


def extract_project_sections(soup):
    sections = {}

    body = soup.find('div', class_=lambda x: x and 'b-detail__body' in x)
    if not body:
        body = soup

    subtitle_nodes = body.find_all('div', class_=lambda x: x and 'b-detail__subtitle' in x)

    for subtitle in subtitle_nodes:
        label = " ".join(subtitle.get_text(" ", strip=True).split())
        if not label:
            continue

        subtitle_cell = subtitle.find_parent('div', class_=lambda x: x and 'grid__cell' in x)
        if subtitle_cell:
            content_cell = subtitle_cell.find_next_sibling('div', class_=lambda x: x and 'grid__cell' in x)
        else:
            content_cell = None

        if not content_cell:
            continue

        content_wrapper = content_cell.find('div', class_=lambda x: x and 'b-detail__content' in x)
        if not content_wrapper:
            content_wrapper = content_cell

        lines = []
        list_items = content_wrapper.find_all('li')
        if list_items:
            for li in list_items:
                text = " ".join(li.get_text(" ", strip=True).split())
                if text:
                    lines.append(f"- {text}")
        else:
            paragraphs = content_wrapper.find_all('p')
            if paragraphs:
                for p in paragraphs:
                    text = " ".join(p.get_text(" ", strip=True).split())
                    if text:
                        lines.append(text)
            else:
                raw_text = " ".join(content_wrapper.get_text(" ", strip=True).split())
                if raw_text:
                    lines.append(raw_text)

        links = []
        for a in content_wrapper.find_all('a'):
            href = a.get('href')
            if not href:
                continue
            link_text = " ".join(a.get_text(" ", strip=True).split())
            links.append({
                "text": link_text if link_text else None,
                "url": href
            })

        section_key = normalize_key(label)
        entry = {
            "label": label,
            "text": "\n".join(lines)
        }
        if links:
            entry["links"] = links

        sections[section_key] = entry

    return sections


def extract_project_detail(soup, listing_name=None, project_url=None):
    project_data = {
        "project": listing_name,
        "project_url": project_url,
        "title": None,
        "metadata": {},
        "sections": {}
    }

    title_h1 = soup.find('h1', class_=lambda x: x and 'b-detail__title' in x)
    if title_h1:
        project_data["title"] = " ".join(title_h1.get_text(" ", strip=True).split())
    elif listing_name:
        project_data["title"] = listing_name

    project_data["metadata"] = extract_project_header_metadata(soup)
    project_data["sections"] = extract_project_sections(soup)

    return project_data


def extract_publication_links(soup, base_url):
    publications = []
    seen_urls = set()

    link_lists = soup.find_all('ul', class_=lambda x: x and 'list-links' in x)
    for ul in link_lists:
        year_heading = ul.find_previous_sibling(['h2', 'h3'])
        year = " ".join(year_heading.get_text(" ", strip=True).split()) if year_heading else None

        items = ul.find_all('li', class_=lambda x: x and 'list-links__item' in x)
        for li in items:
            detail_link = li.find('a', class_=lambda x: x and 'list-links__link' in x)
            if not detail_link:
                detail_link = li.find('a', href=True)
            if not detail_link:
                continue

            raw_href = detail_link.get('href')
            if not raw_href:
                continue

            detail_url = urljoin(base_url, raw_href)
            if detail_url in seen_urls:
                continue

            full_text = " ".join(li.get_text(" ", strip=True).split())
            citation = re.sub(r'\bdetail\b\s*$', '', full_text, flags=re.IGNORECASE).strip(' ,;')

            publications.append({
                "year": year,
                "citation": citation if citation else None,
                "url": detail_url
            })
            seen_urls.add(detail_url)

    return publications


def extract_publication_detail(soup, listing_citation=None, publication_url=None, listing_year=None):
    publication_data = {
        "publication": listing_citation,
        "publication_url": publication_url,
        "year": listing_year,
        "title": None,
        "metadata": {},
        "sections": {}
    }

    title_h1 = soup.find('h1', class_=lambda x: x and 'b-detail__title' in x)
    if title_h1:
        publication_data["title"] = " ".join(title_h1.get_text(" ", strip=True).split())
    elif listing_citation:
        publication_data["title"] = listing_citation

    publication_data["metadata"] = extract_project_header_metadata(soup)
    publication_data["sections"] = extract_project_sections(soup)

    return publication_data


def publication_file_stem(publication_title, fallback_name=None):
    return sanitize_filename(publication_title or fallback_name or 'publication')

def scrape_project_data(config, save_folder):
    save_folder = os.path.join(save_folder, "projects")
    os.makedirs(save_folder, exist_ok=True)
    for item in config:
        name = item['name']
        url = item['url']
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        project_links = []
        link_lists = soup.find_all('ul', class_='list-links')
            
        # 2. Iterate through the list items
        for ul in link_lists:
            items = ul.find_all('li', class_=lambda x: x and 'list-links__item' in x)
            
            for li in items:
                link = li.find('a')
                
                if link:
                    project_name = link.get_text(strip=True)
                    raw_href = link.get('href')
                    
                    if project_name and raw_href:
                        absolute_url = urljoin(url, raw_href)
                        
                        project_links.append({
                            "name": project_name,
                            "url": absolute_url
                        })
        for entry in tqdm(project_links, desc=f"Scraping projects from {name}"):
            project_name, project_url = entry["name"], entry["url"]
            save_path = os.path.join(save_folder, f"{sanitize_filename(project_name)}.json")

            if os.path.exists(save_path):
                print(f"Project '{project_name}' already exists. Skipping...")
                continue

            detail_response = requests.get(project_url)
            detail_url = detail_response.url
            detail_soup = BeautifulSoup(detail_response.content, 'html.parser')

            save_dict = extract_project_detail(
                detail_soup,
                listing_name=project_name,
                project_url=detail_url
            )
            # print(save_dict)
            # exit()
            save_json(save_dict, save_path)

def scrape_publication_data(config, save_folder):
    save_folder = os.path.join(save_folder, "publications")
    os.makedirs(save_folder, exist_ok=True)
    for item in config:
        name = item['name']
        url = item['url']
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        publication_links = extract_publication_links(soup, url)

        for entry in tqdm(publication_links, desc=f"Scraping publications from {name}"):
            listing_citation = entry["citation"]
            publication_url = entry["url"]
            listing_year = entry.get("year")

            detail_response = requests.get(publication_url)
            detail_url = detail_response.url
            detail_soup = BeautifulSoup(detail_response.content, 'html.parser')

            save_dict = extract_publication_detail(
                detail_soup,
                listing_citation=listing_citation,
                publication_url=detail_url,
                listing_year=listing_year
            )
            file_stem = publication_file_stem(
                save_dict.get("title"),
                fallback_name=listing_citation
            )
            save_path = os.path.join(save_folder, f"{file_stem}.json")

            if os.path.exists(save_path):
                print(f"Publication '{file_stem}' already exists. Skipping...")
                continue

            save_json(save_dict, save_path)
        

if __name__ == "__main__":
    args = parser.parse_args()
    config = load_yaml(args.config_path)
    save_folder = os.path.join(args.output_folder, args.config_path.split('/')[-1].split('.')[0])
    os.makedirs(save_folder, exist_ok=True)
    if "programs" in config:
        # scrape_program_data(config.programs, save_folder)
        # scrape_course_data(config.programs, save_folder)
        pass
    # scrape_personnel_links(save_folder)
    # scrape_group_data(config.groups, save_folder)
    # scrape_project_data(config.projects, save_folder)
    scrape_publication_data(config.publications, save_folder)