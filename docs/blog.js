import marked from 'marked';

class BlogManager {
    constructor() {
        this.posts = [];
        this.currentCategory = 'all';
        this.initializeEventListeners();
        this.loadPosts();
    }

    initializeEventListeners() {
        // 分類按鈕點擊事件
        document.querySelectorAll('.category').forEach(button => {
            button.addEventListener('click', (e) => {
                document.querySelectorAll('.category').forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                this.currentCategory = button.dataset.category;
                this.filterPosts();
            });
        });

        // 關閉模態框點擊事件
        window.addEventListener('click', (e) => {
            const postModal = document.getElementById('postModal');
            if (e.target === postModal) {
                postModal.style.display = 'none';
            }
        });
    }

    async loadPosts() {
        // 載入文章
        this.posts = [
            {
                id: 1,
                title: '2024 AI 發展趨勢回顧與 2025 展望',
                content: `
# 2024 AI 發展趨勢回顧與 2025 展望

## 2024 年重要 AI 發展里程碑

### 1. 大型語言模型的演進
- GPT-5 的發布帶來更強的理解能力和更低的幻覺產生
- Claude 3 系列展現出優異的推理能力
- Gemini 2.0 在多模態處理上取得突破

### 2. AI 應用普及化
- 企業 AI 應用成熟度提升
- 個人 AI 助理普及
- 開源 AI 模型蓬勃發展

### 3. AI 法規與倫理
- 歐盟 AI Act 正式實施
- 各國相繼建立 AI 監管框架
- AI 安全與倫理準則逐漸成形

## 2025 年 AI 發展趨勢預測

### 1. 專業化 AI 助理
- 垂直領域特化模型興起
- 醫療、法律等專業領域 AI 應用成熟
- 個人化 AI 教育助理普及

### 2. AI 基礎設施優化
- 新型 AI 晶片架構
- 更節能的模型訓練方法
- 邊緣運算 AI 應用增加

### 3. AI 創作工具革新
- 更精準的文本到圖像/視頻生成
- AI 輔助程式開發更加成熟
- 創意產業 AI 工具鏈完善

### 4. 隱私與安全強化
- 聯邦學習技術普及
- 隱私計算技術成熟
- AI 安全防護機制完善

## 結論

2024 年 AI 技術快速發展，應用場景不斷擴大。展望 2025 年，AI 將朝向更專業化、安全化和普及化發展，為各行業帶來更大的變革。企業和個人都需要積極擁抱 AI 技術，同時關注其帶來的挑戰與機遇。`,
                category: 'machine-learning',
                date: '2024-03-20',
                tags: ['AI', '趨勢分析', '機器學習', '2024回顧', '2025展望']
            },
            {
                id: 2,
                title: 'AI 在企業轉型中的實踐與挑戰',
                content: `
# AI 在企業轉型中的實踐與挑戰

## 企業採用 AI 的關鍵領域

### 1. 客戶服務
- AI 客服系統優化
- 智能推薦引擎
- 客戶行為分析

### 2. 營運效率
- 流程自動化
- 預測性維護
- 庫存管理優化

### 3. 決策支援
- 數據分析與預測
- 風險評估
- 市場趨勢分析

## 實施挑戰與解決方案

### 1. 技術整合
- 系統相容性
- 數據質量
- 基礎設施升級

### 2. 人才培育
- AI 技能培訓
- 組織文化調整
- 變革管理

### 3. 成本控制
- ROI 評估
- 資源分配
- 階段性實施

## 未來展望

企業 AI 轉型是一個持續的過程，需要全面的規劃和執行。成功的關鍵在於找到適合的應用場景，並且能夠有效管理變革過程。`,
                category: 'machine-learning',
                date: '2024-03-21',
                tags: ['AI', '企業轉型', '數位化', '管理']
            }
        ];
        this.displayPosts(this.posts);
    }

    displayPosts(posts) {
        const blogGrid = document.querySelector('.blog-grid');
        if (!blogGrid) return;

        blogGrid.innerHTML = '';
        posts.forEach(post => {
            const article = this.createPostElement(post);
            blogGrid.appendChild(article);
        });
    }

    createPostElement(post) {
        const article = document.createElement('article');
        article.className = 'blog-post';
        
        // 只顯示內容預覽（前150個字符）
        const contentPreview = post.content.substring(0, 150) + '...';
        
        article.innerHTML = `
            <div class="post-header">
                <div class="post-category">
                    <i class="fas ${this.getCategoryIcon(post.category)}"></i>
                    ${post.category}
                </div>
                <h2 class="post-title">${post.title}</h2>
                <div class="post-meta">
                    <span><i class="far fa-calendar"></i> ${post.date}</span>
                </div>
            </div>
            <div class="post-preview">
                ${contentPreview}
            </div>
            <button class="read-more">閱讀更多</button>
            <div class="post-tags">
                ${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
            </div>
        `;

        // 添加閱讀更多按鈕的點擊事件
        const readMoreBtn = article.querySelector('.read-more');
        readMoreBtn.addEventListener('click', () => {
            this.openPostModal(post);
        });

        return article;
    }

    getCategoryIcon(category) {
        const icons = {
            'machine-learning': 'fa-brain',
            'cybersecurity': 'fa-shield-alt',
            'programming': 'fa-code',
            'others': 'fa-ellipsis-h'
        };
        return icons[category] || 'fa-file-alt';
    }

    filterPosts() {
        const filteredPosts = this.currentCategory === 'all' 
            ? this.posts 
            : this.posts.filter(post => post.category === this.currentCategory);
        this.displayPosts(filteredPosts);
    }

    // 添加開啟文章模態框的方法
    openPostModal(post) {
        const postModal = document.getElementById('postModal');
        postModal.innerHTML = `
            <div class="modal-content post-modal-content">
                <div class="modal-header">
                    <h2>${post.title}</h2>
                    <button class="close-modal">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="post-meta">
                        <span><i class="far fa-calendar"></i> ${post.date}</span>
                        <span class="post-category">
                            <i class="fas ${this.getCategoryIcon(post.category)}"></i>
                            ${post.category}
                        </span>
                    </div>
                    <div class="post-content markdown-content">
                        ${marked.parse(post.content)}
                    </div>
                    <div class="post-tags">
                        ${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                    </div>
                </div>
            </div>
        `;

        // 添加關閉按鈕事件
        const closeBtn = postModal.querySelector('.close-modal');
        closeBtn.addEventListener('click', () => {
            postModal.style.display = 'none';
        });

        // 點擊模態框外部關閉
        postModal.addEventListener('click', (e) => {
            if (e.target === postModal) {
                postModal.style.display = 'none';
            }
        });

        postModal.style.display = 'block';
    }
}

// 當 DOM 載入完成後初始化
document.addEventListener('DOMContentLoaded', () => {
    new BlogManager();
}); 