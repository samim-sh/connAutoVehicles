from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests

# req = f'https://141.ir/triptimefiles'
# options = webdriver.ChromeOptions()
# driver = webdriver.Chrome('./chromedriver', options=options)

# driver.get(req)
# driver.find_element(By.ID, "yearBtn0").click()
# WebDriverWait(driver,10).until(EC.element_to_be_clickable((By.ID,"div.triptimeRight button.btn.btn-danger.x[id='yearBtn0']"))).click()
for y in [95,96,97,98,99]:
	for m0, m1 in zip(['01','02','03','04','05','06','07','08','09',10,11,12], [1,2,3,4,5,6,7,8,9,10,11,12]):
		try:
			r = requests.get(f'https://141.ir/storage/otffiles/{y}_{m0}_MonthlyReport_13{y}_{m1}_آزادراه_بومهن_به_تهران_.xlsx')
			output = open(f'Bomehen-Tehran{y}-{m1}.xlsx', 'wb')
			output.write(r.content)
			output.close()
			print(f'done - got it - {y, m1}')
		except:
			continue
