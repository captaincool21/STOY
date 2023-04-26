#define BUZZER 13

char incomingByte;
bool recvData=0;

void setup() {
  Serial.begin(9600);
  pinMode(BUZZER,OUTPUT);
  digitalWrite(BUZZER,LOW);      
}

void loop() {
  if (Serial.available() > 0) {
    incomingByte = Serial.read();
    recvData = 1;
  }

  if(recvData)
  {
    recvData=0;
    if(incomingByte == 'S')
    {
      digitalWrite(BUZZER,HIGH);
      delay(3000);
      digitalWrite(BUZZER,LOW);      
      incomingByte=0;
    }
  }
}
