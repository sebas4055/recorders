import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Email details
smtp_server = "smtp.gmail.com"
smtp_port = 587
sender_email = "weightsend46@gmail.com"  # Replace with your Gmail address
receiver_email = "ricerecorders@gmail.com"  # Replace with the recipient's email
password = "vfnw ybzl sfuk mgay"  # Use the app password you generated

# Email content
subject = "Room Data Update"
body = "Here is the latest room data update attached."

# Create the email message
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = subject

# Attach the body text
msg.attach(MIMEText(body, 'plain'))

# Attach the CSV file (replace the path as needed)
filename = "/home/test/pysense/py/total.csv"  # Replace with the full path to your total.csv
try:
    with open(filename, "rb") as attachment:
        # Create the file attachment part
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename={filename}")
        msg.attach(part)
except Exception as e:
    print(f"Failed to read the file: {e}")

# Send the email using Gmail's SMTP server
try:
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Secure the connection
        server.login(sender_email, password)  # Log in to your Gmail account
        server.sendmail(sender_email, receiver_email, msg.as_string())  # Send the email
    print(f"Email sent to {receiver_email} successfully!")
except Exception as e:
    print(f"Failed to send email: {e}")

