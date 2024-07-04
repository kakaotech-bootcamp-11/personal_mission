# Overview

미션 주제: RDBMS를 활용한 데이터베이스 설계 및 최적화

학습 계획 정보는 아래와 같음

### Day 1

- [x] 문제 정의: RDBMS를 활용한 데이터베이스 설계 및 최적화의 필요성 및 목적 정의
- [x] 솔루션 도출: MySQL을 활용한 데이터베이스 설계 및 최적화 방안 도출
- [x] 설계: 데이터베이스 스키마 및 CRUD 작업 설계
- [x] MySQL 설치 및 초기 설정
- [x] 데이터베이스 및 테이블 생성
- [x] 데이터 삽입을 위한 스키마 설계 및 구현

### Day 2

- [x] CRUD 작업 구현: 데이터 생성(Create), 읽기(Read), 업데이트(Update), 삭제(Delete) 기능 구현
- [x] 인덱싱 구현: 성능 향상을 위한 인덱스 설정
- [ ] 복잡한 쿼리 작성 및 최적화
- [ ] 데이터 모델링: 유연한 스키마 설계 및 변경
- [ ] 트랜잭션 처리 및 데이터 일관성 유지 방안 적용

### Day 3

- [ ] 성능 테스트: 데이터 삽입, 조회, 업데이트, 삭제 작업의 성능 측정
- [ ] 성능 분석: 테스트 결과를 바탕으로 성능 병목점 파악 및 최적화
- [ ] 데이터 백업 및 복구 방안 설계 및 구현
- [ ] 사용자 인터페이스(UI) 구현: 데이터베이스 조작을 위한 간단한 웹 인터페이스 구축
- [ ] 보안 설정: 접근 제어 및 인증 설정

### Day 4

- [ ] 코드 리뷰 및 피드백 반영
- [ ] 기능 시연 영상 제작: 주요 기능 시연 및 설명
- [ ] 성능 테스트 및 분석 결과 보고서 작성
- [ ] README 파일 작성: 사용 방법 및 설정 방법 명시
- [ ] 최종 결과물 제출: 소스 코드, 설정 파일, 보고서, 시연 영상, README 파일



# 🔎진행 및 완료 사항

## ❓문제 정의: RDBMS를 활용한 데이터베이스 설계 및 최적화의 필요성 및 목적 정의

날이 갈수록 애플리케이션은 보다 더 많은 데이터를 수용해야 하고 그러한 데이터를 효율적으로 저장하고 검색할 수 있는 강력한 데이터베이스 시스템에 의존할 수밖에 없다. 그러나 이러한 데이터베이스 시스템 중에서도 관계형 데이터베이스 관리 시스템(RDBMS)은 데이터의 무결성, 일관성, 안정성 등을 보장하며, 다양한 비즈니스 요구사항을 충족시킬 수 있는 강력한 기능을 제공하여 많이 선택되는 데이터베이스 방식 중 하나이다.

하지만, 데이터베이스의 성능이 떨어지면 전체 시스템의 성능에도 큰 영향을 미치기 때문에 데이터베이스의 설계 및 최적화는 매우 중요한 작업이다.

RDBMS를 활용한 데이터베이스 설계 및 최적화의 필요성은 다음과 같은 것들을 고려할 수 있다:

1. 데이터의 효율적인 저장 및 검색:
   - 데이터베이스는 대량의 데이터를 효율적으로 저장하고, 필요한 데이터만을 빠르게 검색할 수 있어야 한다. 잘못된 설계나 최적화가 덜 된 데이터베이스는 데이터 검색 시 불필요한 자원 소비와 시간을 소모한다. 이를 방지하기 위해서는 적절한 스키마 설계와 최적화 기법을 적절히 적용해야 한다.

![image](https://github.com/speardragon/kakaotech-bootcamp/assets/79521972/d4cd0940-6698-4689-95ca-54a2c6fd91bf)

1. **시스템 성능 향상**:
   - 데이터베이스 성능이 전체 시스템 성능에 큰 영향을 미치므로, 데이터베이스의 설계 및 최적화는 시스템 성능을 극대화하는 데 필수적이다. 빠르고 효율적인 데이터베이스는 애플리케이션의 응답 시간을 단축시키고, 사용자 경험을 향상시킬 수 있다.
2. **데이터 일관성 및 무결성 보장**:
   - 관계형 데이터베이스는 데이터의 일관성과 무결성을 보장하기 위한 다양한 제약 조건과 트랜잭션 관리 기능을 제공한다. 이러한 기능들을 효과적으로 활용하기 위해서는 데이터베이스의 설계가 잘 되어 있어야 하며, 데이터의 일관성과 무결성을 유지하기 위한 최적화가 필요하다.
3. **확장성**:
   - 데이터베이스 시스템은 데이터가 증가함에 따라 확장 가능해야 한다. 잘 설계된 데이터베이스는 데이터의 양이 늘어나도 성능 저하 없이 운영될 수 있으며, 필요에 따라 수평적 혹은 수직적으로 확장할 수 있다. 이를 위해서는 초기 설계 단계에서부터 확장성을 고려한 데이터베이스 구조가 필요하다.
4. **복잡한 비즈니스 로직 구현**:
   - 많은 애플리케이션은 복잡한 비즈니스 로직을 구현해야 한다. 이러한 로직은 데이터베이스의 스키마와 밀접하게 연관되어 있으며, 최적화된 데이터베이스 설계를 통해 효율적으로 구현될 수 있다. 이를 통해 애플리케이션의 성능을 극대화할 수 있다.

따라서, RDBMS를 활용한 데이터베이스 설계 및 최적화의 목적은 단순히 데이터를 저장하는 것에서 그치지 않고, 데이터의 보다 효율적인 관리, 시스템의 성능을 극대화, 데이터의 일관성과 무결성을 유지하는 것들을 모두 수반한다고 볼 수 있다.

이를 통해 전체 시스템의 안정성과 성능을 보장할 수 있으며 이번 미션에서는 RDBMS 중 가장 대표주자라고 할 수 있는 MySQL을 활용하여 앞서 서술한 목적들을 달성하기 위해 데이터베이스 설계 및 최적화 방안을 모색하고, 실질적인 구현을 통해 그 효과를 검증해 보는 것이 목적이다.



## 💡솔루션 도출: MySQL을 활용한 데이터베이스 설계 및 최적화 방안 도출

효율적이고 최적화된 데이터베이스 시스템을 구축하기 위해서는 MySQL을 활용한 다양한 설계 방식 혹은 최적화 방안을 고려해야 한다.

MySQL은 관계형 데이터베이스 관리 시스템(RDBMS) 중 하나로, 높은 성능과 안정성을 제공하면서도 비교적 사용이 용이하여 널리 사용되는 기술 중 하나이다. 이번 미션에서는 MySQL을 활용하여 데이터베이스 설계 및 최적화를 통해 성능을 극대화하는 방안을 도출하고자 한다.



### **1. 데이터베이스 스키마 설계**

- 정규화(Normalization):
  - 데이터 중복을 최소화하고 데이터 무결성을 유지하기 위해 데이터베이스를 정규화한다. 정규화 과정을 통해 데이터를 여러 테이블로 나누어 각 테이블이 하나의 주제에 집중하도록 한다.
  - 예를 들어, 고객 정보와 주문 정보를 별도의 테이블로 분리하여 관리한다.
- 비정규화(Denormalization):
  - 정규화된 테이블이 많아질 경우, 조인 연산이 빈번해져 성능 저하가 발생할 수 있다. 이러한 경우, 성능 향상을 위해 일부 데이터를 비정규화하여 중복 저장할 수 있다.
  - 예를 들어, 주문 테이블에 고객 이름을 포함시켜 조인 연산을 줄인다.
- ERD(Entity-Relationship Diagram) 작성:
  - 데이터베이스 설계의 기초 단계로, ERD를 작성하여 테이블 간의 관계를 명확히 한다. 이를 통해 데이터 구조를 시각적으로 이해하고 설계 오류를 사전에 방지할 수 있다.



### **2. 인덱싱(Indexing)**

- 기본 인덱스 생성:
  - 자주 조회되는 컬럼에 인덱스를 생성하여 검색 속도를 향상시킨다. 기본 인덱스는 B-Tree 구조를 사용하여 데이터 접근을 빠르게 한다.
  - 예를 들어, 고객 테이블의 '고객 ID' 컬럼에 인덱스를 생성한다.
- 복합 인덱스(Composite Index):
  - 여러 컬럼을 조합한 복합 인덱스를 생성하여 복잡한 쿼리의 성능을 최적화한다.
  - 예를 들어, '고객 ID'와 '주문 날짜' 컬럼을 조합한 복합 인덱스를 생성한다.
- 인덱스 유지보수:
  - 인덱스는 데이터를 삽입, 수정, 삭제할 때마다 갱신되므로, 불필요한 인덱스를 제거하여 성능을 유지한다.



### **3. 쿼리 최적화(Query Optimization)**

- 쿼리 작성 시 최적화 기법 적용:
  - 쿼리를 작성할 때, 쿼리 실행 계획을 분석하여 최적화 기법을 적용한다. 쿼리 실행 계획은 EXPLAIN 명령어를 통해 확인할 수 있다.
  - 예를 들어, 불필요한 조인을 제거하고, 서브쿼리를 JOIN으로 대체하는 등의 방법을 사용한다.
- 캐싱(Cache) 활용:
  - 자주 사용되는 쿼리 결과를 캐싱하여 데이터베이스 부하를 줄인다. MySQL의 Query Cache 기능을 활용하거나, 애플리케이션 레벨에서 캐싱을 구현할 수 있다.
- 준비된 쿼리(Prepared Statement) 사용:
  - 준비된 쿼리를 사용하면, 쿼리 컴파일과 최적화 과정을 한 번만 수행하고, 이후에는 컴파일된 쿼리를 재사용할 수 있어 성능이 향상된다.



### **4. 데이터베이스 설정 및 튜닝**

- 서버 설정 최적화:
  - MySQL 서버의 설정 파일(my.cnf)을 조정하여 메모리 사용량, 캐시 크기, 연결 수 등을 최적화한다.
  - 예를 들어, InnoDB 버퍼 풀 크기, 쿼리 캐시 크기, 연결 타임아웃 등을 조정한다.
- 트랜잭션 설정:
  - 트랜잭션 격리 수준을 적절히 설정하여 데이터 일관성을 유지하면서도 성능을 최적화한다. 기본적으로 InnoDB 스토리지 엔진의 경우 REPEATABLE READ 격리 수준을 사용한다.
- 로그 관리:
  - 슬로우 쿼리 로그를 활성화하여 실행 시간이 긴 쿼리를 모니터링하고, 이를 최적화한다.
  - 예를 들어, 1초 이상의 실행 시간이 소요되는 쿼리를 로그에 기록하고 분석하여 성능을 향상시킨다.



### **5. 성능 모니터링 및 분석**

- 모니터링 도구 활용:
  - MySQL 자체의 성능 스키마(Performance Schema)와 외부 모니터링 도구를 활용하여 실시간으로 데이터베이스 성능을 모니터링한다.
  - 예를 들어, MySQL Workbench, Percona Monitoring and Management(PMM) 등을 사용하여 성능 지표를 분석한다.
- 주기적인 성능 테스트:
  - 주기적으로 성능 테스트를 수행하여 데이터베이스의 성능 상태를 점검하고, 필요시 최적화 작업을 수행한다.
  - 예를 들어, 데이터베이스 부하 테스트 도구를 사용하여 대량의 데이터 삽입, 조회, 수정, 삭제 작업을 시뮬레이션한다.



### **6. 데이터 모델링 및 스키마 변경**

- 유연한 데이터 모델링:
  - 비즈니스 요구사항 변화에 유연하게 대응할 수 있도록 데이터 모델링을 설계한다.
  - 예를 들어, 테이블 간의 관계를 유연하게 설정하고, 스키마 변경이 용이하도록 설계한다.
- 스키마 변경 시 영향 분석:
  - 스키마 변경이 필요한 경우, 변경 사항이 기존 시스템에 미치는 영향을 분석하고, 변경 작업을 신중하게 수행한다.
  - 예를 들어, 테이블 구조 변경 시 데이터 마이그레이션 계획을 세우고, 변경 작업을 단계적으로 수행한다.

이와 같은 다양한 설계 및 최적화 방안을 통해 MySQL 데이터베이스의 성능을 극대화하고, 안정적이고 효율적인 데이터 관리를 실현할 수 있다. 이번 프로젝트에서는 이러한 방안을 실제로 적용하여 데이터베이스를 설계하고 최적화하며, 그 효과를 검증할 것이다.



## 설계: 데이터베이스 스키마 및 CRUD 작업 설계

이번 프로젝트에서는 간단한 블로그 시스템을 만든다고 가정하고 데이터베이스 스키마와 CRUD 작업을 설계해보겠습니다. 블로그 시스템은 다음과 같은 주요 기능을 포함합니다: 사용자 관리, 게시글 작성 및 조회, 댓글 작성 및 조회. 이러한 기능을 구현하기 위해 데이터베이스 스키마를 설계하고, 각 기능에 필요한 CRUD 작업을 정의합니다.



### **데이터베이스 스키마 설계**

1. **사용자(User) 테이블**
   - 사용자 정보를 저장하는 테이블입니다.
   - 주요 컬럼: 사용자 ID(user_id), 사용자 이름(username), 이메일(email), 비밀번호(password), 가입일자(created_at)
2. **게시글(Post) 테이블**
   - 블로그 게시글 정보를 저장하는 테이블입니다.
   - 주요 컬럼: 게시글 ID(post_id), 제목(title), 내용(content), 작성자 ID(user_id), 작성일자(created_at), 수정일자(updated_at)
3. **댓글(Comment) 테이블**
   - 게시글에 대한 댓글 정보를 저장하는 테이블입니다.
   - 주요 컬럼: 댓글 ID(comment_id), 내용(content), 작성자 ID(user_id), 게시글 ID(post_id), 작성일자(created_at)



### **CRUD 작업 설계**

1. 사용자(User) 관리
   - Create (사용자 등록):
     - 새로운 사용자를 등록한다.
     - 요구사항: 사용자 이름, 이메일, 비밀번호를 입력받아 새로운 사용자를 생성한다. 이메일은 고유해야 하며, 비밀번호는 안전하게 저장되어야 한다.
   - Read (사용자 조회):
     - 특정 사용자의 정보를 조회한다.
     - 요구사항: 사용자 ID나 이메일을 입력받아 해당 사용자의 정보를 반환한다.
   - Update (사용자 정보 수정):
     - 기존 사용자의 정보를 수정한다.
     - 요구사항: 사용자 이름, 이메일, 비밀번호를 수정할 수 있으며, 이메일은 고유성을 유지해야 한다.
   - Delete (사용자 삭제):
     - 특정 사용자를 삭제한다.
     - 요구사항: 사용자 ID를 입력받아 해당 사용자를 삭제한다. 관련된 게시글 및 댓글도 함께 삭제된다.
2. 게시글(Post) 관리
   - Create (게시글 작성):
     - 새로운 게시글을 작성한다.
     - 요구사항: 제목, 내용, 작성자 ID를 입력받아 새로운 게시글을 생성한다. 작성자는 등록된 사용자여야 한다.
   - Read (게시글 조회):
     - 게시글 목록 및 특정 게시글의 상세 정보를 조회한다.
     - 요구사항: 모든 게시글의 목록을 조회할 수 있으며, 특정 게시글 ID를 입력받아 해당 게시글의 제목, 내용, 작성자 정보 및 작성일자를 반환한다.
   - Update (게시글 수정):
     - 기존 게시글을 수정한다.
     - 요구사항: 게시글 ID와 새로운 제목, 내용을 입력받아 해당 게시글을 수정한다. 수정일자를 갱신한다.
   - Delete (게시글 삭제):
     - 특정 게시글을 삭제한다.
     - 요구사항: 게시글 ID를 입력받아 해당 게시글을 삭제한다. 관련된 댓글도 함께 삭제된다.
3. 댓글(Comment) 관리
   - Create (댓글 작성):
     - 특정 게시글에 댓글을 작성한다.
     - 요구사항: 댓글 내용, 작성자 ID, 게시글 ID를 입력받아 새로운 댓글을 생성한다. 작성자와 게시글은 각각 등록된 사용자와 게시글이어야 한다.
   - Read (댓글 조회):
     - 특정 게시글에 대한 댓글 목록을 조회한다.
     - 요구사항: 게시글 ID를 입력받아 해당 게시글에 달린 모든 댓글의 목록을 반환한다. 각 댓글의 내용, 작성자 정보 및 작성일자를 포함한다.
   - Update (댓글 수정):
     - 기존 댓글을 수정한다.
     - 요구사항: 댓글 ID와 새로운 내용을 입력받아 해당 댓글을 수정한다. 수정일자는 갱신하지 않는다(최초 작성일만 유지).
   - Delete (댓글 삭제):
     - 특정 댓글을 삭제한다.
     - 요구사항: 댓글 ID를 입력받아 해당 댓글을 삭제한다.



## MySQL 설치 및 초기 설정

이 작업에서는 MySQL 설치, 보안 설정, 초기 사용자 및 더미 데이터베이스 생성 등의 작업을 포함한다.

먼저 시작하기에 앞서 실습을 진행한 노트북 환경은 다음과 같다.

- MacBook Pro 14
- macOS: Sonoma 14.5
- 일시: 2024-07-02
- 터미널: iTerm2
- MySQL: 8.3.0



### [Step 1] 터미널을 켠다.

![image](https://github.com/speardragon/kakaotech-bootcamp/assets/79521972/78aa7740-9eca-41e7-97c9-ae5d5fb98989)



### [Step 2] brew install mysql

`brew install mysql` 명령어를 입력하여 컴퓨터에 MySQL을 설치한다.

- 만약 homebrew가 설치되어 있지 않다면? → https://brew.sh/ko/

설치를 완료하면 아래와 같은 문구들이 나온다.

![image](https://github.com/speardragon/kakaotech-bootcamp/assets/79521972/7eca5056-a721-48c7-be39-2d14926db5eb)

Caveats(주의 사항)을 보면 root 계정 비밀번호 설정 없이 mysql만 설치되어 경고를 하고 있다.

이를 해결하기 전에 먼저 다음 사진과 같이 `brew services start mysql` 명령어를 통해 mysql을 실행시킨다.

- 해당 명령어를 통해서는 시작프로세스에 등록하여 컴퓨터가 부팅 시에 mysql도 자동으로 실행해 주는 역할을 한다.

  ![image](https://github.com/speardragon/kakaotech-bootcamp/assets/79521972/aedf31ce-795a-4f72-9196-3c7e12847670)

그 다음, 아래 안내된 명령어인 `mysql_secure_installation`을 통해 설정을 마쳐 mysql 설정을 완료할 수 있다.



- mysql_secure_installation

  해당 명령어를 통해선 비밀번호를 설정하고 초기에 세팅되어 있던 익명 유저, 더미 테이블 등을 삭제할지 등을 물어본다.

  나는 개인적으로 가이드를 따라 웬만하면 yes 선택지를 선택했지만 기호에 따라 설정을 마치면 된다.

  **1. 비밀번호 복잡도 검사**

  VALIDATE PASSWORD COMPONENT can be used to test passwords

  and improve security. It checks the strength of password

  and allows the users to set only those passwords which are

  secure enough.

  Would you like to setup VALIDATE PASSWORD component?

  Press y|Y for Yes, any other key for No

  비밀번호 복잡도 검사를 적용할 것인지 설정하는 과정입니다. 복잡도 검사를 적용하게 된다면 다음과 같은 비밀번호 조건이 붙습니다.

  - 최소 8자리 이상의 비밀번호
  - 최소 1개의 영문자
  - 최소 1개의 특수문자
  - 최소 1개의 숫자

  간단하게 비밀번호를 설정해 두실 분들은 No로 진행하면 됩니다.

  이후 설정할 비밀번호 입력이 나오는데 입력해도 보이지 않는 것이 정상입니다.

  ***

  **2. 익명의 사용자 삭제**

  By default, a MySQL installation has an anonymous user,

  allowing anyone to log into MySQL without having to have

  a user account created for them. This is intended only for

  testing, and to make the installation go a bit smoother.

  You should remove them before moving into a production

  environment.

  Remove anonymous users? (Press y|Y for Yes, any other key for No)

  익명의 사용자를 삭제할 것인지 물어봅니다.

  설치 과정에서 테스트와 설치를 좀 더 원활하게 진행하기 위해 생성되었던 계정인데 Yes로 삭제해 줍니다.

  ***

  **3. root 계정 원격 접속 차단**

  Normally, root should only be allowed to connect from

  'localhost'. This ensures that someone cannot guess at

  the root password from the network.

  Disallow root login remotely? (Press y|Y for Yes, any other key for No)

  root 계정을 원격(외부)에서의 접속을 차단할 것인지 묻습니다. 일반적인 경우라면 root 계정은 원격 접속은 차단해 두는 것이 권장됩니다. Yes로 차단을 하여 주시면 됩니다. 원격작업이 필요한 경우는 root 말고 db 계정을 따로 생성하여 원격 권한을 주는 것이 가장 바람직합니다.

  ***

  **4. 테스트 데이터베이스 삭제**

  By default, MySQL comes with a database named 'test' that

  anyone can access. This is also intended only for testing,

  and should be removed before moving into a production

  environment.

  Remove test database and access to it? (Press y|Y for Yes, any other key for No)

  테스트 데이터베이스를 삭제할 것인지 물어봅니다.

  필요 없으니 과감히 Yes로 버려주셔도 됩니다.

  ***

  **5. 현재까지의 변경사항 즉시 적용**

  Reloading the privilege tables will ensure that all changes

  made so far will take effect immediately.

  Reload privilege tables now? (Press y|Y for Yes, any other key for No)

  Yes를 통해 변경사항을 즉시 적용시켜 줍니다.



### [Step 3] mysql 로그인

이제 설치가 되었으니 기본 설정이 완료된 root 계정으로 로그인을 할 수 있다.

- ```
  mysql -u root -p
  ```

  - -u 뒤에는 접속할 계정의 이름이 오며, -p 옵션은 해당 계정의 비밀번호를 이용하여 접속하겠다는 의미이다.

![image](https://github.com/speardragon/kakaotech-bootcamp/assets/79521972/cf56df3b-8c49-45ea-b289-ccc191e5ed6a)

초기에 설정한 비밀번호를 맞게 입력하면 터미널에 `myslq>` 에 커서가 깜빡이는 것을 볼 수 있고 이는 Mysql 터미널에 들어왔음을 의미한다.

만약 mysql 터미널을 종료하고 싶다면 `exit`을 입력하면 된다. mysql 서버 자체를 종료하고 싶다면 먼저 mysql 터미널이 종료된 상태에서 `mysql.server stop`을 입력하면 된다.



## 데이터베이스 및 테이블 생성

이제 mysql을 사용할 수 있게 되었으니 데이터베이스 및 테이블을 생성할 수 있게 되었다.

### 데이터베이스 보기

```
SHOW DATABASES;
```

이 명령어는 현재 mysql 서버에 생성되어 있는 모든 데이터베이스들을 볼 수 있는 명령어이다.

반드시 세미콜론을 붙여주어야 명령어가 정상적으로 실행되며, 실수로 붙이지 못한 경우 그 다음 입력을 세미콜론만 입력 후 엔터를 눌러주면 명령어가 실행될 수 있다.

<img src="https://github.com/speardragon/kakaotech-bootcamp/assets/79521972/0ec97a93-6522-42a1-896c-1a0435de2e47" alt="image" style="zoom: 33%;" />

위에 이미 생성된 데이터베이스들은 기본적으로 생성되어 있는 것들이며, 주로 설정, 계정 등과 관련된 정보를 담은 데이터베이스로 현재 Mysql의 설정이나 계정 및 권한이 궁금하다면 저 중 알맞은 데이터베이스로 쿼리를 날려 정보를 확인할 수 있다.



### 데이터베이스 생성

```
CREATE DATABASES (데이터베이스 이름)
```

이 명령어는 새로운 데이터베이스를 하나 생성하는 명령어이다.



### 데이터베이스 사용

데이터베이스를 만들기만 했다고 바로 사용할 수 있는 것은 아니다. 내가 특정 데이터베이스를 사용하겠다고 알리고 쿼리를 날려야 해당 데이터베이스의 정보를 가져올 수 있다.

```
use (데이터베이스 이름)
```



### 블로그를 위한 데이터베이스 생성 및 데이터베이스 사용

<img src="https://github.com/speardragon/kakaotech-bootcamp/assets/79521972/d0922c53-172d-4fbd-a375-0bab3fbc325d" alt="image" style="zoom:50%;" />

각 명령어를 입력해 blogDB라는 이름의 데이터베이스를 만들고 사용 명령어까지 입력한다.



### 데이터베이스 테이블 생성

블로그 시스템을 위해 사용자(User), 게시글(Post), 댓글(Comment) 테이블을 생성한다.



1. **사용자(User) 테이블 생성**:

   - 사용자 정보를 저장하는 테이블을 생성한다.

     ```sql
     CREATE TABLE User (
       user_id INT AUTO_INCREMENT PRIMARY KEY,
       username VARCHAR(50) NOT NULL,
       email VARCHAR(100) NOT NULL UNIQUE,
       password VARCHAR(255) NOT NULL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
     );
     ```



2. **게시글(Post) 테이블 생성**:

- 블로그 게시글 정보를 저장하는 테이블을 생성한다.

  ```sql
  CREATE TABLE Post (
    post_id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    user_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES User(user_id)
  );
  ```



3. **댓글(Comment) 테이블 생성:**

- 게시글에 대한 댓글 정보를 저장하는 테이블을 생성한다.

  ```sql
  CREATE TABLE Comment (
    comment_id INT AUTO_INCREMENT PRIMARY KEY,
    content TEXT NOT NULL,
    user_id INT,
    post_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES User(user_id),
    FOREIGN KEY (post_id) REFERENCES Post(post_id)
  );
  ```



## 데이터 삽입을 위한 스키마 설계 및 구현

이제 데이터 삽입을 위해 각 테이블 별로 스키마를 세부적으로 설계하고 실제로 삽입까지 해볼 것이다.



### **사용자(User) 테이블 스키마 설계**

- **목적**: 사용자 정보를 저장하고 관리하기 위한 테이블입니다.

- 컬럼:

  - **`user_id`**: 정수형, 자동 증가, 기본 키
  - **`username`**: 문자열, 사용자 이름
  - **`email`**: 문자열, 고유, 이메일 주소
  - **`password`**: 문자열, 암호화된 비밀번호
  - **`created_at`**: 타임스탬프, 사용자가 가입한 날짜와 시간



### **게시글(Post) 테이블 스키마 설계**

- **목적**: 블로그 게시글 정보를 저장하고 관리하기 위한 테이블입니다.

- 컬럼:

  - **`post_id`**: 정수형, 자동 증가, 기본 키
  - **`title`**: 문자열, 게시글 제목
  - **`content`**: 텍스트, 게시글 내용
  - **`user_id`**: 정수형, 작성자 ID, 외래 키 (User 테이블 참조)
  - **`created_at`**: 타임스탬프, 게시글 작성 날짜와 시간
  - **`updated_at`**: 타임스탬프, 게시글 수정 날짜와 시간



### **댓글(Comment) 테이블 스키마 설계**

- **목적**: 게시글에 대한 댓글 정보를 저장하고 관리하기 위한 테이블입니다.

- 컬럼:

  - **`comment_id`**: 정수형, 자동 증가, 기본 키
  - **`content`**: 텍스트, 댓글 내용
  - **`user_id`**: 정수형, 작성자 ID, 외래 키 (User 테이블 참조)
  - **`post_id`**: 정수형, 게시글 ID, 외래 키 (Post 테이블 참조)
  - **`created_at`**: 타임스탬프, 댓글 작성 날짜와 시간



### 데이터베이스 ERD 설계

<img width="952" alt="image" src="https://github.com/kakaotech-bootcamp-11/personal_mission/assets/79521972/fad8da0d-2c96-4084-bb4b-cb215659fde1">



### **데이터 삽입 스키마 구현**

이제 각 테이블에 데이터를 삽입하는 SQL 스크립트를 작성하여 데이터베이스에 데이터를 삽입합니다.

**1. 사용자(User) 데이터 삽입**

```sql
-- 사용자 데이터 삽입
INSERT INTO User (username, email, password) VALUES
('alice', 'alice@example.com', 'password123'),
('bob', 'bob@example.com', 'password456'),
('charlie', 'charlie@example.com', 'password789');
```



**2. 게시글(Post) 데이터 삽입**

```sql
-- 게시글 데이터 삽입
INSERT INTO Post (title, content, user_id) VALUES
('First Post', 'This is the content of the first post', 1),
('Second Post', 'This is the content of the second post', 2),
('Third Post', 'This is the content of the third post', 1);
```



**3. 댓글(Comment) 데이터 삽입**

```sql
-- 댓글 데이터 삽입
INSERT INTO Comment (content, user_id, post_id) VALUES
('This is a comment on the first post', 2, 1),
('Another comment on the first post', 3, 1),
('Comment on the second post', 1, 2);
```



### **데이터 확인**

데이터가 제대로 삽입되었는지 확인하기 위해 각 테이블의 데이터를 조회한다.

```sql
-- 사용자 데이터 조회
SELECT * FROM User;

-- 게시글 데이터 조회
SELECT * FROM Post;

-- 댓글 데이터 조회
SELECT * FROM Comment;
```

![image](https://github.com/speardragon/kakaotech-bootcamp/assets/79521972/127635f3-cb03-45c2-ba39-fb728515d3d5)

이 과정을 통해 데이터 삽입, 조회, 수정, 삭제와 같은 CRUD 작업을 수행할 수 있는 기반을 마련하였다.



# CRUD 작업 구현: 데이터 생성(Create), 읽기(Read), 업데이트(Update), 삭제(Delete)

이 단계에서는 블로그 시스템의 사용자(User), 게시글(Post), 댓글(Comment) 테이블에 대해 데이터 생성(Create), 읽기(Read), 업데이트(Update), 삭제(Delete) 작업을 구현합니다. 각 작업에 대한 요구사항을 중심으로 설명하고, 필요한 SQL 쿼리와 예제를 포함합니다.



#### 1. 사용자(User) 관리

**Create (사용자 등록)**

- **요구사항**: 새로운 사용자를 등록한다. 사용자 이름, 이메일, 비밀번호를 입력받아 새로운 사용자를 생성한다. 이메일은 고유해야 하며, 비밀번호는 안전하게 저장되어야 한다.

- SQL 쿼리:

  ```sql
  INSERT INTO User (username, email, password) VALUES ('alice', 'alice@example.com', 'password123');
  ```



**Read (사용자 조회)**

- **요구사항**: 특정 사용자의 정보를 조회한다. 사용자 ID나 이메일을 입력받아 해당 사용자의 정보를 반환한다.

- SQL 쿼리:

  ```
  -- 사용자 ID로 조회
  SELECT * FROM User WHERE user_id = 1;
  
  -- 이메일로 조회
  SELECT * FROM User WHERE email = 'alice@example.com';
  ```



**Update (사용자 정보 수정)**

- **요구사항**: 기존 사용자의 정보를 수정한다. 사용자 이름, 이메일, 비밀번호를 수정할 수 있으며, 이메일은 고유성을 유지해야 한다.

- SQL 쿼리:

  ```
  UPDATE User SET username = 'Alice Wonderland', email = 'alice.w@example.com', password = 'newpassword123' WHERE user_id = 1;
  ```



**Delete (사용자 삭제)**

- **요구사항**: 특정 사용자를 삭제한다. 사용자 ID를 입력받아 해당 사용자를 삭제한다. 관련된 게시글 및 댓글도 함께 삭제된다.

- SQL 쿼리:

  ```
  DELETE FROM User WHERE user_id = 1;
  ```



#### 2. 게시글(Post) 관리

**Create (게시글 작성)**

- **요구사항**: 새로운 게시글을 작성한다. 제목, 내용, 작성자 ID를 입력받아 새로운 게시글을 생성한다. 작성자는 등록된 사용자여야 한다.

- SQL 쿼리:

  ```
  INSERT INTO Post (title, content, user_id) VALUES ('First Post', 'This is the content of the first post', 1);
  ```



**Read (게시글 조회)**

- **요구사항**: 게시글 목록 및 특정 게시글의 상세 정보를 조회한다. 모든 게시글의 목록을 조회할 수 있으며, 특정 게시글 ID를 입력받아 해당 게시글의 제목, 내용, 작성자 정보 및 작성일자를 반환한다.

- SQL 쿼리:

  ```
  -- 모든 게시글 조회
  SELECT * FROM Post;
  
  -- 특정 게시글 조회
  SELECT * FROM Post WHERE post_id = 1;
  ```



**Update (게시글 수정)**

- **요구사항**: 기존 게시글을 수정한다. 게시글 ID와 새로운 제목, 내용을 입력받아 해당 게시글을 수정한다. 수정일자를 갱신한다.

- SQL 쿼리:

  ```
  UPDATE Post SET title = 'Updated First Post', content = 'This is the updated content of the first post' WHERE post_id = 1;
  ```



**Delete (게시글 삭제)**

- **요구사항**: 특정 게시글을 삭제한다. 게시글 ID를 입력받아 해당 게시글을 삭제한다. 관련된 댓글도 함께 삭제된다. 관리자거나 게시글을 올린 사용자만 삭제할 수 있다.

- SQL 쿼리:

  ```
  DELETE FROM Post WHERE post_id = 1;
  ```



#### 3. 댓글(Comment) 관리

**Create (댓글 작성)**

- **요구사항**: 특정 게시글에 댓글을 작성한다. 댓글 내용, 작성자 ID, 게시글 ID를 입력받아 새로운 댓글을 생성한다. 작성자와 게시글은 각각 등록된 사용자와 게시글이어야 한다.

- SQL 쿼리:

  ```
  INSERT INTO Comment (content, user_id, post_id) VALUES ('This is a comment on the first post', 2, 1);
  ```



**Read (댓글 조회)**

- **요구사항**: 특정 게시글에 대한 댓글 목록을 조회한다. 게시글 ID를 입력받아 해당 게시글에 달린 모든 댓글의 목록을 반환한다. 각 댓글의 내용, 작성자 정보 및 작성일자를 포함한다.

- SQL 쿼리:

  ```
  SELECT * FROM Comment WHERE post_id = 1;
  ```



**Update (댓글 수정)**

- **요구사항**: 기존 댓글을 수정한다. 댓글 ID와 새로운 내용을 입력받아 해당 댓글을 수정한다. 수정일자는 갱신하지 않는다(최초 작성일만 유지).

- SQL 쿼리:

  ```
  UPDATE Comment SET content = 'This is the updated comment' WHERE comment_id = 1;
  ```



**Delete (댓글 삭제)**

- **요구사항**: 특정 댓글을 삭제한다. 댓글 ID를 입력받아 해당 댓글을 삭제한다. 댓글을 단 사용자거나 관리자만 지울 수 있다.

- SQL 쿼리:

  ```
  DELETE FROM Comment WHERE comment_id = 1;
  ```





# 인덱싱 구현: 성능 향상을 위한 인덱스 설정

인덱스는 데이터베이스 성능 향상을 위해 사용되는 중요한 도구이다. 

인덱스는 테이블에서 데이터를 빠르게 검색할 수 있도록 돕는 자료 구조이기 때문에 인덱스를 적절히 사용하면 데이터 검색 속도를 크게 향상시킬 수 있지만, 그렇다고 인덱스를 과도하게 사용하면 데이터 삽입, 삭제, 업데이트 성능이 저하될 수도 있습니다. 



## 1. 인덱스 개요

- **기본 인덱스**: 기본 키(primary key)는 자동으로 인덱스가 생성되며, 유니크 키(unique key)도 인덱스가 생성됩니다.
- **보조 인덱스**: 검색 속도를 향상시키기 위해 추가로 생성하는 인덱스이다.
- **복합 인덱스**: 두 개 이상의 컬럼을 조합한 인덱스입니다.



## 2. 인덱싱 설계

인덱스를 설계할 때는 테이블에서 자주 사용되는 쿼리와 검색 조건을 분석하여 효율적인 인덱스를 설정해야 한다. 블로그 시스템에서 다음과 같은 인덱스를 설계한다:

- User 테이블
  - `email`: 사용자 이메일을 자주 검색하므로 인덱스를 설정합니다.
- Post 테이블
  - `user_id`: 작성자별로 게시글을 검색할 때 유용한다.
  - `title`: 게시글 제목을 자주 검색하는 경우에 유용한다.
- Comment 테이블
  - `user_id`: 사용자별로 댓글을 검색할 때 유용한다.
  - `post_id`: 게시글별로 댓글을 검색할 때 유용한다.



## 3. 인덱스 설정 SQL

각 테이블에 대한 인덱스를 설정하는 SQL 쿼리는 다음과 같습니다:

```sql
-- 데이터베이스 사용
USE blogDB;

-- User 테이블에 인덱스 설정
CREATE INDEX idx_user_email ON User (email);

-- Post 테이블에 인덱스 설정
CREATE INDEX idx_post_user_id ON Post (user_id);
CREATE INDEX idx_post_title ON Post (title);

-- Comment 테이블에 인덱스 설정
CREATE INDEX idx_comment_user_id ON Comment (user_id);
CREATE INDEX idx_comment_post_id ON Comment (post_id);
```



이 보고서는 블로그 시스템의 데이터베이스 성능 향상을 위한 인덱스 설계 및 구현 과정에 대해 설명합니다. 인덱스는 데이터 검색 속도를 크게 향상시키며, 블로그 시스템의 사용자, 게시글, 댓글 데이터를 효율적으로 관리할 수 있도록 돕습니다.



#### 2. 인덱스 설계 원칙

- **검색 속도 향상**: 자주 검색되는 컬럼에 인덱스를 설정하여 데이터 검색 속도를 향상시킵니다.
- **데이터 무결성 유지**: 기본 키와 유니크 키는 자동으로 인덱스가 생성되며, 데이터의 무결성을 유지합니다.
- **적절한 인덱스 사용**: 과도한 인덱스 사용을 지양하고, 데이터 삽입, 삭제, 업데이트 성능 저하를 방지합니다.



#### 3. 인덱스 설정

1. **User 테이블 인덱스 설정**

   - **목적**: 사용자 이메일로 빠르게 검색하기 위해 인덱스를 설정합니다.

   - 쿼리:

     ```
     CREATE INDEX idx_user_email ON User (email);
     ```

2. **Post 테이블 인덱스 설정**

   - **목적**: 작성자별로 게시글을 검색하고, 게시글 제목으로 검색할 때 성능을 향상시키기 위해 인덱스를 설정합니다.

   - 쿼리:

     ```
     CREATE INDEX idx_post_user_id ON Post (user_id);
     CREATE INDEX idx_post_title ON Post (title);
     ```

3. **Comment 테이블 인덱스 설정**

   - **목적**: 사용자별로 댓글을 검색하고, 게시글별로 댓글을 검색할 때 성능을 향상시키기 위해 인덱스를 설정합니다.

   - 쿼리:

     ```
     CREATE INDEX idx_comment_user_id ON Comment (user_id);
     CREATE INDEX idx_comment_post_id ON Comment (post_id);
     ```



## 4. 인덱스 적용

#### 1. DataGrip에서 데이터베이스 연결

1. **DataGrip 실행**:
   - DataGrip을 실행한다.
2. **데이터베이스 연결 설정**:
   - 왼쪽 상단의 `Database` 탭을 클릭한다.
   - `+` 버튼을 클릭하고, `Data Source` > `MySQL`을 선택한다.
   - `Host`, `Port`, `User`, `Password` 등을 입력하여 MySQL 데이터베이스에 연결한다.
   - 연결 테스트(Test Connection) 버튼을 눌러 연결이 성공적으로 이루어졌는지 확인한다.
   - `OK` 버튼을 눌러 연결을 저장한다.



#### 2. SQL 콘솔 열기

연결된 데이터베이스를 오른쪽 클릭하고, `Jump to Data Source` > `Console` > `New Console`을 선택하여 SQL 콘솔을 엽니다.



#### 3. 인덱스 설정 SQL 실행

아래 SQL 쿼리를 SQL 콘솔에 입력하고 실행하여 인덱스를 설정한다.

```
-- 데이터베이스 사용
USE blogDB;

-- User 테이블에 인덱스 설정
CREATE INDEX idx_user_email ON User (email);

-- Post 테이블에 인덱스 설정
CREATE INDEX idx_post_user_id ON Post (user_id);
CREATE INDEX idx_post_title ON Post (title);

-- Comment 테이블에 인덱스 설정
CREATE INDEX idx_comment_user_id ON Comment (user_id);
CREATE INDEX idx_comment_post_id ON Comment (post_id);
```



#### 4. SQL 쿼리 실행

1. **SQL 쿼리 입력**:
   - 위의 SQL 쿼리를 SQL 콘솔 창에 복사하여 붙여넣는다.
2. **쿼리 실행**:
   - 쿼리를 선택한 상태에서 `Ctrl + Enter` 키를 눌러 실행한다.
   - 쿼리가 성공적으로 실행되면, 아래쪽의 `Output` 창에서 결과를 확인할 수 있다.



#### 5. 인덱스 확인

- 데이터베이스 구조를 탐색하여 `User`, `Post`, `Comment` 테이블을 찾아본다.
- 각 테이블을 클릭하여 테이블 구조를 확인하고, 설정된 인덱스를 확인한다.



혹은 인덱스를 확인할 수 있는 쿼리를 통해 각 테이블마다의 상세 정보를 확인하여 Index 설정 관련 정보를 볼 수 있다.

```
SELECT
    TABLE_NAME,
    INDEX_NAME,
    COLUMN_NAME,
    SEQ_IN_INDEX,
    NON_UNIQUE,
    INDEX_TYPE
FROM
    information_schema.STATISTICS
WHERE
    TABLE_SCHEMA = 'blogDB'
ORDER BY
    TABLE_NAME, INDEX_NAME, SEQ_IN_INDEX;
```



<img width="1020" alt="image" src="https://github.com/speardragon/kakaotech-bootcamp/assets/79521972/86663361-10fe-4f80-a1e7-1123d2843157">



#### 6. 성능 테스트

1. **성능 테스트 시나리오 설정**:

   - 인덱스를 설정하기 전과 후의 쿼리 실행 시간을 비교합니다.
   - 예를 들어, 사용자 이메일로 검색하는 쿼리를 실행하여 실행 시간을 비교한다.

2. **성능 테스트 쿼리**:

   ```
   -- 사용자 이메일로 검색
   SELECT * FROM User WHERE email = 'alice@example.com';
   
   -- 작성자별로 게시글 검색
   SELECT * FROM Post WHERE user_id = 1;
   
   -- 게시글 제목으로 검색
   SELECT * FROM Post WHERE title = 'First Post';
   
   -- 게시글별로 댓글 검색
   SELECT * FROM Comment WHERE post_id = 1;
   ```

3. **쿼리 실행 및 시간 측정**:

   - 각 쿼리를 실행하고, 실행 시간을 기록합니다.
   - 인덱스 설정 전과 후의 실행 시간을 비교하여 인덱스의 성능 향상 효과를 확인한다.





# 복잡한 쿼리 작성 및 최적화





# 데이터 모델링: 유연한 스키마 설계 및 변경





# 트랜잭션 처리 및 데이터 일관성 유지 방안 적용













# ❗️작업 중 발생한 이슈/해결 방법

## MySQL 설치

![image](https://github.com/speardragon/kakaotech-bootcamp/assets/79521972/6869a82a-bbef-4330-ab69-4144e25776d8)

homebrew를 재설치 해주었더니 해결됨

```bash
/bin/bash -c "$(curl -fsSL <https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh>)"
```









# 새롭게 알게 된 내용

.



# Reference

.



# Takeaway

.
